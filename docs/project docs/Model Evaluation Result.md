# Re-ID模型评估结果

## 预训练Resnet-18模型直接用于人车对

| Dataset    | Rank-1 | Rank-5 | Rank-10 | mAP  | mINP | metric |
| :--------- | :----- | :----- | :------ | :--- | :--- | :----- |
| BikePerson | 0.12   | 0.65   | 1.22    | 0.19 | 0.09 | 0.15   |


## VehicleID模型直接用于人车对

| Dataset    | Rank-1 | Rank-5 | Rank-10 | mAP  | mINP | metric |
| :--------- | :----- | :----- | :------ | :--- | :--- | :----- |
| BikePerson | 0.97   | 2.47   | 3.71    | 0.60 | 0.13 | 0.78   |

## Market1501 People Re-ID模型直接用于人车对

| Dataset    | Rank-1 | Rank-5 | Rank-10 | mAP  | mINP | metric |
| :--------- | :----- | :----- | :------ | :--- | :--- | :----- |
| BikePerson | 0.93   | 2.55   | 3.80    | 0.62 | 0.13 | 0.77   |

## Market1501 People Re-ID模型训练后用于人车对

| Dataset    | Rank-1 | Rank-5 | Rank-10 | mAP   | mINP  | metric |
| :--------- | :----- | :----- | :------ | :---- | :---- | :----- |
| BikePerson | 67.95  | 81.73  | 88.68   | 66.71 | 51.75 | 67.33  |

## Market1501 People Re-ID模型训练后用于自行车

| Dataset    | Rank-1 | Rank-5 | Rank-10 | mAP   | mINP  | metric |
| :--------- | :----- | :----- | :------ | :---- | :---- | :----- |
| BikePerson | 56.25  | 71.43  | 79.53   | 52.79 | 34.39 | 54.52  |


