# alkaloid-api
## API for prediction of alkaloids from NMR

### This repo contains a trained XGBoost model for the prediction of alkaloids. The input information is 13C NMR spectra and the prediction is an alkaloid or non-alkaloid molecule.



### 1. Dataset
The data originally comes from the [Naproc-13 Dataset](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00293). It was extracted and joined into a csv file where a label was attached to each spectra. The used labels are: 1 if it corresponds to an alkaloid and 0 if it doesn't. Each array from the dataset is padded with -999 for obtaining fixed sized arrays of length equal to the max number of spectra for an alkaloid (37). 

### 2. Data preprocessing

For the experiments, two resampling techniques were tested for alliviating the data imbalance (0.8 % positives):

1. Random undersampling
2. Combination of random undersampling and oversampling 

The selected new imbalance ratio was 10% of positives. The imblearn implementation of the SMOTEENN algorithm was used for the resampling process.

### 3. Hyperparameters Search

Selected hyperparameters for obtaining the best model:

  * max depth
  * alpha
  * lambda
  * min child weight
  * number of estimators
  * random seed
  * eta

The search was performed using Bayes Optimization minimizing the -recall score (`experiments/hyperparam_search.ipynb`). The hyperopt package was employed for the algorithm search implementation. 

Best hyperparmeters:

|hyperparameter|value|
|--------------|-----|
|'eta'|0.43911010940117146|
|'max_depth'|15|
|'min_child_weight'|2|
|'n_estimators'|120|
|'reg_alpha'|0.0747528345435633|
|'reg_lambda'| 5.147611893010373|
|'seed'| 0 |

For training reproduction install the requirements in the root dir:

`pip install -r requirements.txt`

then in the `train` dir run: 

`python train.py`

and see the artifacts information from the MLflow UI:

`mlflow ui` 

and open the browser at the localhost http://127.0.0.1:5000.

### 4. Evaluation Metrics

The dataset was divided into 3 sets: Train, test, and validation sets. The selected size ratios were Train-Test: 0:70, Validation set: 0.30 

The hyperparameter search was performed over the Train-test set using Cross-Validation 5-fold. The metric considered to evaluate the model performance was recall.

For the resampling strategy 1 (Random undersampling), the best CV metric is:

* 0.41

And for the resampling strategy 2:

* 0.96 

However, for the validation set, the achieved metrics are

* 0.12

and

* 0.31

for the resampling strategies 1, and 2, respectively. Thus, the selected model was XGBoost with the best CV-5 hyperparameters and resampling stategy 2. Complementary metrics for this model (validation set) are:

|Metric|Value|
|------|-----|
|precision|0.22|
|f1-score|0.26|
|roc auc|0.65|

Given the metrics results, one must be cautious about the practical usage of this model to predict alkaloids.

### 5. Prediction

You can use this repo for your own predicitions using 13C NMR data. 

### 5.1 Locally from cloned repo
(Requires Python >= 3.8)

First, clone the repo via

`git clone https://github.com/saulhazelius/alkaloid-api`

or download and unzip the zip file from the github site.

Then, within the `project` dir install the dependencies:

`pip install -r requirements_predict.txt`

In order to make predictions, modify the `spectra.json` file with your own NMR input values.
Finally, run:

`python predict.py`

to perform the alkaloid prediction.

### 5.2 Locally with aws sam invoke function
(5.2 and 5.3 Requires Docker and AWS SAM CLI)

You can use `sam build` to build the Docker image. Inside the dir `api` you can find the `template.yml` and the `Dockerfile` for building and the `app.py` and `events/event.json` for making predictions. Also the trained model `model.xgb` is contained. Thus, in the dir `api` execute:

`sam build` 

Then, modify the `events/event.json` with your NMR info following the established format and test the prediction with:

`sam local invoke InferenceFunction --event events/event.json`.

### 5.3 Local API

Before deploying to AWS ECR and invoking the Lambda InferenceFunction using an endpoint, test locally the api with a simulation of the API Gateway:

`sam local start-api`

and sent a request with your own NMR data directly. The local api uses the port 3000 in the localhost and the NMR values can be written as: `{"NMR1": 100.0, "NMR2": 50.0}`. In this case the command would be   

`curl -X POST http://127.0.0.1:3000/classification -d '{"NMR1": 100.0, "NMR2": 50.0}'`

then you should see the response:

`{"predicted_label": 0, "alkaloid_presence": false}`


### 5.4 As a microservice with API Gateway
In order to deploy the image you can build the container image, create the ECR repository, push the image to the repository, and perform the deployment. First, build the cointainer with:

`sam build --use-container`

Then, create the ECR repository named `alkaloid-repo`:
aws ecr create-repository --repository-name alkaloid-repo
The created repository contains an uri that can be retreieved with:

`aws ecr describe-repositories --repository-name alkaloid-repo`

Use the repository uri to login:

`aws ecr get-login-password --region <your_region> | docker login -u AWS --password-stdin <repository_uri>`

Tag and push:

`docker tag alkaloidfunction:python3.7-v1 <repository_uri>`

`docker push <repository_uri>`

Now, you can deploy the image creating the CloudFormation stack named `alkaloidfunction`:

`sam deploy --stack-name alkaloidfunction --region <your_region> --image-repository <repository_uri> --capabilities CAPABILITY_IAM`

After that, test the created endpoint using curl or Postman:

`curl -X POST -H "Content-Type: application/json" -d @../project/spectra.json <endpoint>`

Finally, delete the created resources:

`aws cloudformation delete-stack --stack-name alkaloidfunction`

