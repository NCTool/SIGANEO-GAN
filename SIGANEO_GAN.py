import os

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import warnings

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.hidden = 32
        self.code_len = 32
        self.encoder = nn.Sequential(
            nn.Linear(in_features=15, out_features=self.hidden),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden, out_features=self.code_len),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.code_len, out_features=self.hidden),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden, out_features=15),
            nn.Sigmoid()
        )
        self.modelName = 'SIGANEO-GAN'

    def forward(self, x):
        # Split the data and the mask matrix.
        x_data = x[:, :int(x.shape[1] / 2)]
        x_mask = x[:, int(x.shape[1] / 2):]
        x_noise = Normal(0, 1).sample(sample_shape=x_data.size()).cuda()

        # Add initial random data for missing values.
        x_tmp = x_data * x_mask + x_noise * (1 - x_mask)
        code = self.encoder(x_tmp)
        x_pie = self.decoder(code)
        return code, x_pie


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hidden = 32
        self.code_len = 64
        self.encoder = nn.Sequential(
            nn.Linear(in_features=15, out_features=self.hidden),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.hidden, out_features=self.code_len),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.code_len, out_features=self.hidden),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.hidden, out_features=15),
            nn.LeakyReLU()
        )
        self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=1)

    def forward(self, x, mask=None):
        code = self.encoder(x)
        x_pie = self.decoder(code)

        if mask is not None:
            x_noise = torch.abs(Normal(0, 0.5).sample(sample_shape=mask.size()).cuda())
            x_hint = self.imputeData(mask, mask, x_noise)
            x_pie = x_pie * x_hint

        x_pie = self.conv(x_pie.view(-1, 1, 1, x_pie.shape[1]))
        x_pie = x_pie.squeeze(2).permute(0, 2, 1)

        return code, x_pie


class Imputer:
    def __init__(self, filePath=None):
        self.data = self._getData(filePath)
        self.model = Generator()

    def _getData(self, filePath=None):
        print('Reading data...')
        file = pd.read_csv(filePath)

        rawData = file[
            ['Gene_FPKM', 'RNA_ref_read_count', 'RNA_alt_read_count', 'Gene_TPM', 'ah1', 'ah2',
             'MHCFlurry_MT_Aff', 'MHCFlurry_MT_Rank', 'netMHCpan_Aff', 'netMHCpan_Rank',
             'netMHCstabpan_Thalf(h)',
             'netMHCstabpan_Rank', 'MATHLA_Aff', 'R_score', 'VAF']]

        return self._normalizeData(rawData)

    def _normalizeData(self, data):
        print('Normalizing data...')
        data = self._changeBA(data)
        data_mask = (~pd.isna(data)).astype(int)
        tmp = np.log10(data.astype(float) + 0.0001) * data_mask
        data = tmp
        data[data_mask == 0] = np.nan
        return data

    def _changeBA(self, df):
        col_list = ['MHCFlurry_MT_Aff', 'MHCFlurry_MT_Rank', 'netMHCpan_Aff', 'netMHCpan_Rank',
                    'netMHCstabpan_Thalf(h)',
                    'netMHCstabpan_Rank', 'MATHLA_Aff']
        ba_result = np.array(df[col_list].fillna(-99999999))
        ba_mask = (ba_result != -99999999).astype(int)
        for i in range(ba_result.shape[1]):
            if i == 4:
                # Restore the value of netMHCstabpan_Thalf.
                ba_result[:, i] = 2 ** -ba_result[:, i]
            else:
                temp_ba_result = ba_result[:, i] + (-(ba_mask[:, i] - 1) * ba_result[:, i].max())
                if i in [0, 2, 6]:
                    # Convert binding affinity value.
                    ba_result[:, i] = self._from_ic50(temp_ba_result).reshape(-1) * ba_mask[:, i]
                elif i in [1, 3, 5]:
                    # Convert rank value.
                    ba_result[:, i] = (1 - (ba_result[:, i] / 100)) * ba_mask[:, i]
            ba_result[:, i][ba_mask[:, i] == 0] = np.nan

        for i in range(len(col_list)):
            df[col_list[i]] = ba_result[:, i]
        return df

    def _from_ic50(self, ic50, max_ic50=50000.0):
        try:
            x = 1.0 - (np.log(np.abs(ic50)) / np.log(max_ic50))
        except RuntimeWarning:
            x = 1

        if max_ic50 == 50000.0:
            return np.minimum(1.0, np.maximum(0.0, x)).reshape(-1, 1)
        else:
            x = np.nan_to_num(x, neginf=0, posinf=0, nan=0)
            x[ic50 < 0] *= -1
            x[np.abs(ic50) <= 1] = ic50[np.abs(ic50) <= 1]
            return x

    def _getDataLoader(self, x, batch_size=128):
        x = np.array(x, dtype='float')
        x_tenser = torch.Tensor.float(torch.from_numpy(x))
        dataset = TensorDataset(x_tenser)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=False)
        return dataloader

    def _maskedDataLoader(self, data):
        mask = data.isna()
        data = data.fillna(0)
        data = np.hstack((np.array(data), np.array(~mask).astype(int)))
        return self._getDataLoader(data, batch_size=256)

    def _test(self, dataloader, isCuda=True):
        self.model.load_state_dict(torch.load('model/{}.pt'.format(self.model.modelName)))

        # Switch to CPU or GPU mode according to the environment.
        if torch.cuda.is_available() and isCuda:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        pred = None

        with torch.no_grad():
            for b_x in tqdm(dataloader):
                self.model.eval()
                b_x = b_x[0]

                if torch.cuda.is_available() and isCuda:
                    b_x = b_x.cuda()

                temp = self.model(b_x)[1].cpu().numpy()
                temp = temp.reshape(temp.shape[0], temp.shape[1])

                if pred is None:
                    pred = temp
                else:
                    pred = np.vstack((pred, temp))

        return pred

    def imputedDATA(self):
        print('Imputing...')
        TEST_dl = self._maskedDataLoader(self.data)
        ba_fusion = self._test(dataloader=TEST_dl)
        raw_data = np.array(self.data).astype(float)
        mask = (~(np.isnan(raw_data))).astype(int)      # Create a mask matrix
        raw_data = np.nan_to_num(raw_data, 0)       # To avoid affecting calculations, None values are temporarily filled with 0.

        # Preserve the original data according to the mask, and imputing new data.
        imputedData = raw_data * mask + ba_fusion * (1 - mask)
        imputedData = pd.DataFrame(imputedData)
        imputedData.columns = list(self.data.columns)

        return imputedData


if __name__ == '__main__':
    fileName = 'toyData.csv'
    imp = Imputer(f'data/raw_data/{fileName}')                                  # Input
    imputedData = imp.imputedDATA()

    imputedData.to_csv(f'data/gan_data/{fileName}', index=None)      # Output

