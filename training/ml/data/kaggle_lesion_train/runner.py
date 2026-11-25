# Kaggle runner (appended by make.sh): unzip the built dataset, train, keep best.pt / last.pt / summary.json.
if __name__ == "__main__" and Path("/kaggle/input").exists():
    import subprocess, zipfile
    z = next(Path("/kaggle/input").rglob("lesion_ds.zip"))
    zipfile.ZipFile(z).extractall("/kaggle/temp")
    qc("/kaggle/temp/lesion_ds")  # ml/data/lesion_qc.py
    open("/kaggle/temp/cfg.yaml", "w").write(CFG)
    sys.argv = ["train", "/kaggle/temp/cfg.yaml", "--data", "/kaggle/temp/lesion_ds", "--out", "/kaggle/working/L1"]
    main()
