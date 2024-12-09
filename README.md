# Continuous X - ML

This project explores the building of a Continuous X (CI/CD, Continuous Training and Continuous Monitoring) pipeline for a Machine Learning app.

## 1. Background
### Continuous X Principles
### Continuous Integration
#### Containerization
### Continuous Delivery and Deployment
### Continuous Training
### Continuous Monitoring

## 2. Implementation
### Overview
The main goal of this project is to build a full pipeline for a machine learning app to continuously train new model modifications, continuously test and evaluate the modified model, and, if certain conditions are met, continuously deploy the accepted model. Additionally, continuous monitoring would also be occurring, which is not in the scope of this project.

### Expectations
What is expected to happen with this pipeline is that, if any changes are made to our model and pushed to our remote repository on GitHub, supposing that we collaborate with other teammates on this project, the pipeline will be triggered to retrain with the new changes, then evaluating the model and finally determining whether deployment should happen or not.

1. The `train` script will be triggered. This will run the python script for training the model.
    1. If the da
    2. Item 3b
2. 
### Initial Project Setup
1. Run `reserve_chameleon.ipynb`. This notebook will reserve and configure resources on KVM. This sets up host `node-0`. This only needs to be done in the project.
2. Run `Colab_Chameleon_Ubuntu22.ipynb`. This notebook will be run every time we need to create a new lease for resources for our GPU host because our old one expired or because we do not have one. This will be our `GPU` host.
3. SSH GPU host and node-0
    * Open terminal in GPU host via SSH:
        * Open a terminal window within the Jupyter instance (can also be done locally if local public SSH keys are added to authorized_keys in the host) and run (in my case, but it is also printed on `Colab_Chameleon_Ubuntu22.ipynb`):

        ```
        $ ssh cc@192.5.86.227
        ```
    * Generate SSH keys for the host:
    ```
    $ ssh-keygen -t rsa -b 4096 -C "your@email.edu"
    ```
    
    * Now we need to copy the new public key into node-0's authorized keys. This is needed to have access to node-0 to later send our trained and evaluated models to the other host.
        * Manually, in the GPU terminal, we will copy the output of this command:
        ```
        $ cat /home/cc/.ssh/id_rsa.pub
        ```
        On node-0's terminal (we can open a terminal and similar to the GPU host's instructions to log into its terminal, we would run ``` ssh cc@129.114.25.151 ```, changing the address to whichever `reserve_chameleon.ipynb` outputs), we will paste the previous output and save:

        ```
        $ nano .ssh/authorized_keys
        ```
        * Alternatively, we can just run, with the appropriate permissions, from the GPU host's terminal entering the address of node-0:

        ```
        $ ssh-copy-id cc@129.114.25.151
        ```

Now, we are ready to send data to node-0 from the GPU host and access its terminal as well.

4. Start runners
Our pipeline is built on GitHub actions, which are triggered upon modifications to the model code and pushes to remote. When this occurs, we want the jobs to be run on the hosts that we have set up on points 1 and 2. In order to do so, we need to have our hosts ready to take the jobs that come from GitHub actions, so they need to be _listening_.

You will need to go [here](https://github.com/tin2294/image-classification-continuous-x/settings/actions/runners) and click on `New self-hosted runner` and select `Linux`:

        #### ADD SCREENSHOTS

The site will give you the following commands that tou will need to run on your GPU and node-0 terminals (your token will be different and each host will have a different runner so this will need to be done once per host):

```
# Create a folder
$ mkdir actions-runner && cd actions-runner
# Download the latest runner package
$ curl -o actions-runner-osx-x64-2.321.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-osx-x64-2.321.0.tar.gz
# Optional: Validate the hash
$ echo "b2c91416b3e4d579ae69fc2c381fc50dbda13f1b3fcc283187e2c75d1b173072  actions-runner-osx-x64-2.321.0.tar.gz" | shasum -a 256 -c
# Extract the installer
$ tar xzf ./actions-runner-osx-x64-2.321.0.tar.gz
$ ./config.sh --url https://github.com/tin2294/image-classification-continuous-x --token AGFBHYJYLF2XG2LVC4HGEGLHK4MI6
# Last step, run it!
$ ./run.sh
```
NOTE: for `node-0`, this only needs to be done once and for your GPU host, this needs to be done every time we set up resources.
When we configure the runner, make sure to use the following labels (depending on the host you are configuring the runner for):
* node-0
* gpu-p100

        #### ADD SCREENSHOTS

Now, every time we want to start the runners, on our terminal we need to make sure we are in the `actions-runner` folder and then run `run.sh`:

```
$ cd actions-runner
$ ./run.sh
```

5. If this is your first time setting up the project, Docker volumes will need to be created. The volumes we need on the GPU host are the following:
    1. food11_data
    2. saved_models
    3. model-to-deploy

The commands we need to run on the GPU host's terminal:

```
$ docker volume create food11_data
$ docker volume create saved_models
$ docker volume create model-to-deploy
```

You can check that they have been created with (the output should be the names of the volumes we have created):

```
$ docker volume ls
```


### Initial Deploy and Testing
After the setup, we are ready to deploy.

### Pipeline Run-Through
1. `Train` script:
    1. Docker container
    2. Repo
    3. Dataset
    4. Dependencies
    5. Data organization
    6. Running training python script: model_train.py
    7. Stop container

2. `Evaluate` script:
    1. Start Docker container
    2. Get repo
    3. Dependencies
    4. Extract last model
    5. Run evaluation python script: evaluate_model.py
    6. Stop container

3. `Redeploy` script:
    1. Extract latest model
    2. Build and push Docker image
    3. Set new Docker image for redeployment
    4. Make sure redeployment is complete

### DEMO

## 3. Future Improvements
* Continuous monitoring
* Add more tests
* Set up alerts
* Item 2a
* Item 2b
    * Item 3a
    * Item 3b



## 4. Sources

