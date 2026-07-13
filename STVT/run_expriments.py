import subprocess

experiments = [
    {
        "test_dataset": "7,16,17,23,25",
        "roundtimes": "Rsplit1_original_SK",
    },
    {
        "test_dataset": "2,4,5,7,24",
        "roundtimes": "Rsplit2_original_SK",
    },
    {
        "test_dataset": "4,7,8,11,23",
        "roundtimes": "Rsplit3_original_SK",
    },
    {
        "test_dataset": "3,5,7,8,15",
        "roundtimes": "Rsplit4_original_SK",
    },
    {
        "test_dataset": "1,9,16,17,23",
        "roundtimes": "Rsplit5_original_SK",
    },
]

for exp in experiments:
    cmd = [
        "python",
        "train.py",
        "--dataset", "SumMe_RFR_Normalized",
        "--test_dataset", exp["test_dataset"],
        "--roundtimes", exp["roundtimes"],
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)