"""Active experiment dataset paths, in one place so profileProcessing.py and dataAnalysis.py agree.
Switch datasets by moving the "ACTIVE" pair; PROCESSED_FILE is derived from RAW_FILE. Build the
processed cache once with `python profileProcessing.py` after switching.
"""
_BASE = "HDf5data/RealExperiments/ClayAndWater_2026_05_28"

# Exp3 water-content change:
# RAW_FILE = f"{_BASE}/Exp3/watercontentchangeExp2Sensor.h5"
# PLC_FILE = f"{_BASE}/Exp3/waterContenExp2PLC.csv"

# Exp2 water content:
# RAW_FILE = f"{_BASE}/Exp2/watercontentExperimentSensor_28_05_26.h5"
# PLC_FILE = f"{_BASE}/Exp2/waterContenExperimentPLC_28_05_26.csv"

# Exp1 velocity alterations (ACTIVE):
RAW_FILE = f"{_BASE}/Exp1/velocityAlterationsExperiment28_05_26_sensor.h5"
PLC_FILE = f"{_BASE}/Exp1/velocityAlterationsExpPLC_28_05_26.csv"

PROCESSED_FILE = RAW_FILE.replace(".h5", "_processed.h5")
