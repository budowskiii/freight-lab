import numpy as np


def report(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ape = np.abs(err) / y
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE%": float(100 * ape.mean()),
        "MedAPE%": float(100 * np.median(ape)),
    }
