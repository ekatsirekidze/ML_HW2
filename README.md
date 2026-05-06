# IEEE-CIS Fraud Detection

## კონკურსის მიმოხილვა

ამ პროექტის მიზანია `isFraud` კლასის პროგნოზირება IEEE-CIS Fraud Detection მონაცემებზე.
ეს არის მაღალი სიზუსტის ბინარული კლასიფიკაციის ამოცანა, სადაც ძირითადი შეფასების მეტრიკაა **ROC-AUC**.

- კონკურსი: [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
- DagsHub + MLflow


ეტაპები:

1. **Cleaning** - თავიდან ამოვშალე არასაჭირო სვეტები (ძალიან missing).
2. **Feature Engineering** - TransactionDT/email/frequency/aggregation ბლოკები, raw მონაცემები უფრო ინფორმაციული რომ გამეხადა.
3. **Feature Selection** - რამდენიმე FS სტრატეგია, რომ დამედგინა იძლეოდა თუ არა სარგებელს სვეტების შეკვეცა.
4. **Training + HPO** - თითოეული model family-სთვის ცალკე ექსპერიმენტები ჩატარდა.
5. **Final Pipeline Registration** - საუკეთესო კონფიგურაციები გადავიტანე სრულ `sklearn Pipeline`-ში და დავარეგისტრირე MLflow Model Registry-ში.
6. **Inference Notebook** - რეგისტრირებული მოდელი ჩავტვირთე MLflow-დან და raw test მონაცემზე გავუშვი `predict_proba`.

ამ სტრუქტურამ მომცა რეპროდუცირებადი ექსპერიმენტები და მკაფიო პასუხი კითხვაზე "რომელი მოდელი ჯობია და რატომ".

---

## Feature Engineering და Preprocessing ლოგიკა

`src/` მოდულებში preprocessing/FE დავტოვე reusable transformer-ებად, რომ:
- იგივე ლოგიკამ იმუშაოს train-ზე და test-ზე;
- leakage არ მოხდეს manual data handling-ით;
- inference notebook-ში დამატებითი preprocessing აღარ იყოს საჭირო.

ძირითადი გადაწყვეტილებები:
- numeric სვეტები: constant imputation (`-999`)
- categorical სვეტები: impute + ordinal encode (unknown კატეგორიების დაცვით)
- TransactionDT-დან კალენდარული ნიშნები
- email-domain ნიშნები
- frequency და aggregate ტიპის ნიშნები

შედეგად საბოლოო რეგისტრირებული მოდელი არის ერთი მთლიან pipeline-ად შეფუთული არტეფაქტი.


## Feature Selection სტრატეგია

ვცადე რამდენიმე ვარიანტი:
- ყველა სვეტის დატოვება
- მაღალი კორელაციის სვეტების მოშორება
- top-MI სვეტები
- tree-importance top სვეტები

საბოლოოდ საუკეთესო შედეგები ბევრ model family-ში მივიღე `fs_v1_all`-თან (ანუ ყველა სვეტის დატოვებით), რაც ამ კონკრეტულ ამოცანაზე მოსალოდნელიცაა - tree-based ensemble მოდელები ხშირად თავად ახერხებენ noisy ნიშნების გამკლავებას.


## Training შედეგები (Holdout val_auc)

| Model Family | val_auc |
|---|---:|
| XGBoost | **0.9191487720446068** |
| GradientBoosting | 0.906595386619669 |
| RandomForest | 0.871213451218027 |
| AdaBoost | 0.8641235071982853 |
| Bagging | 0.8465909515566861 |
| LogisticRegression | 0.8444702074168362 |
| MLP | 0.8339559306281703 |
| DecisionTree | 0.7811671583898963 |

### ინტერპრეტაცია

- საუკეთესო შედეგი მიიღო **XGBoost**-მა.
- მეორე ადგილზე **GradientBoosting**-ია, რაც ადასტურებს რომ boosting ამ dataset-ზე ყველაზე ძლიერად მუშაობს.
- `DecisionTree` როგორც single model ყველაზე დაბალია, ხოლო bagging/random forest/boosting ეტაპობრივად აუმჯობესებს ხარისხს.


## საუკეთესო მოდელი

- **Model:** `XGBoost_Fraud_Pipeline`
- `train_auc`: `0.9766074997671338`
- `val_auc`: `0.9191487720446068`
- `gap`: `0.057458727722527025`
- `fit_sec`: `112.55629348754883`

### რატომ ეს მოდელი

- ყველა მოდელს შორის ყველაზე მაღალი holdout AUC ჰქონდა.
- overfit gap არსებობს, მაგრამ კონტროლირებადია და საბოლოო validation შედეგი მაინც საუკეთესოა.
- inference pipeline-ად რეგისტრირდება მთლიანად (cleaning + FE + model), რაც deployment-ს ამარტივებს.


## Kaggle Submission შედეგი

Inference notebook-იდან (`model_inference.ipynb`) დავაგენერირე `submission.csv` და competition-ზე დავასაბმითე.

- **Public LB:** `0.922070`
- **Private LB:** `0.892155`


## MLflow / DagsHub Tracking

ყველა მნიშვნელოვანი run (cleaning, FE, FS, training, CV, final) დაილოგა MLflow-ში.
Model Registry-ში რეგისტრირებულია:

- `LogisticRegression_Fraud_Pipeline`
- `DecisionTree_Fraud_Pipeline`
- `Bagging_Fraud_Pipeline`
- `RandomForest_Fraud_Pipeline`
- `AdaBoost_Fraud_Pipeline`
- `GradientBoosting_Fraud_Pipeline`
- `XGBoost_Fraud_Pipeline`
- `MLP_Fraud_Pipeline`


## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── model_experiment_LogisticRegression.ipynb
│   ├── model_experiment_DecisionTree.ipynb
│   ├── model_experiment_Bagging.ipynb
│   ├── model_experiment_RandomForest.ipynb
│   ├── model_experiment_AdaBoost.ipynb
│   ├── model_experiment_GradientBoosting.ipynb
│   ├── model_experiment_XGBoost.ipynb
│   ├── model_experiment_MLP.ipynb
│   └── model_inference.ipynb
└── src/
    ├── __init__.py
    ├── data.py
    ├── preprocessing.py
    ├── feature_engineering.py
    ├── feature_selection.py
    └── mlflow_utils.py
```