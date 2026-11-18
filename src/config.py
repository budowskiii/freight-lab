"""Final blend configuration selected using time-split validation."""
from src.model import S1_DEFAULT

# Model A: full feature set (my first design)
A = dict(s1_cols=S1_DEFAULT, s2_cols=None)  # s2_cols=None -> every engineered column

# Model D: feature set found by backward elimination + grid-searched hyperparameters
D_S1 = ["equipment", "weight", "market_index", "quote_signal", "t"]
D_S2 = ["distance", "log_dist", "inv_dist", "equip", "market_index", "quote_signal", "mi_dev", "qs_dev",
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
D_PARAMS = dict(num_leaves=31, min_child_samples=30, alpha=0.1, learning_rate=0.03, n_estimators=600,
                subsample=0.8, colsample_bytree=0.8)
D = dict(s1_cols=D_S1, s2_cols=D_S2, lgb_params=D_PARAMS, delta=0.12)
