"""Check temporal preprocessing and submission output contracts."""
import unittest
import numpy as np
import pandas as pd
from src.cv import prepare_fold
from src.data import clean

class PipelineTests(unittest.TestCase):
    def test_fold_imputation_uses_only_training_weights(self):
        frame=pd.DataFrame({
            'date':pd.to_datetime(['2025-01-01','2025-01-02','2025-02-01','2025-02-02']),
            'weight':[100.,np.nan,90000.,np.nan], 'equipment':['Dry Van']*4,
            'market_index':[1.]*4,'quote_signal':[2.]*4,
            'distance':[100.]*4,'pickup_lat':[30.]*4,'pickup_lon':[-90.]*4,
            'delivery_lat':[35.]*4,'delivery_lon':[-85.]*4,
        })
        training,test,features,test_features=prepare_fold(frame,frame.index[:2],frame.index[2:])
        self.assertEqual(training.weight.iloc[1],100.)
        self.assertEqual(test.weight.iloc[1],100.)
        self.assertEqual(test.weight.iloc[0],90000.)
        self.assertEqual(list(features.columns),list(test_features.columns))
        self.assertLess(training.date.max(),test.date.min())

    def test_cleaning_does_not_mutate_input_or_drop_heavy_loads(self):
        source=pd.DataFrame({'date':pd.to_datetime(['2025-01-01']*3),'weight':[-30000.,np.nan,47500.],'equipment':['Dry Van']*3,'market_index':[1.,np.nan,1.]})
        original=source.copy(deep=True)
        result,_=clean(source)
        pd.testing.assert_frame_equal(source,original)
        self.assertEqual(result.weight.iloc[0],30000.)
        self.assertEqual(result.weight.iloc[2],47500.)
        self.assertEqual(len(result),len(source))
        self.assertEqual(result.weight_missing.iloc[1],1)

if __name__=='__main__':unittest.main()
