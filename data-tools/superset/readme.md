# set up Superset on ubuntu 16.04 LTS

![logo](https://superset.incubator.apache.org/_images/s.png)

Apache Superset (incubating) is a modern, enterprise-ready business intelligence web application.
Compared with business-focused BI tool like Tableau, superset is more technology-navy. It supports more types of visualization and able to work in distributed manner to boost the query performance. Most importantly, it is free of charge!

An example dashboard:

![screenshot](https://superset.incubator.apache.org/_images/bank_dash.png)

Let's go and set it up.

## create a virtualenv

Assume Anaconda is installed for python management.

```
# create a virtualenv with python 3.6
conda create -n superset python=3.6
```

## install in virtualenv

enter the virtual environment and follow the [official installation guide](https://superset.incubator.apache.org/installation.html#superset-installation-and-initialization).

```
# enter virtualenv
source activate superset

# install superset follow official installation guide
pip install superset

# Create an admin user (you will be prompted to set username, first and last name before setting a password)
fabmanager create-admin --app superset

# Initialize the database
superset db upgrade

# Load some data to play with
superset load_examples

# Create default roles and permissions
superset init

# To start a development web server on port 8088, use -p to bind to another port
superset runserver -d

```
![isntallation](https://raw.githubusercontent.com/6chaoran/data-story/master/data-tools/superset/superset-installation.png)

You are now in superset debug mode, if you are successfully. Just go to http://localhost:8088 and you will see the login page.
![login-page](https://raw.githubusercontent.com/6chaoran/data-story/master/data-tools/superset/superset-login.png)

After log in, you can view the example dashboards:
![dashboard](https://raw.githubusercontent.com/6chaoran/data-story/master/data-tools/superset/superset-dashboard.png)

## set up systemd service

If you want to deploy superset in production, you have to run it on the background. We can make superset run as systemd service, so that it can run as background proccess and re-start when failed or aborted.

```
cd /etc/systemd/system
sudo touch superset.service
sudo vim superset.service
```

write in following code

```
[Unit]
Description=Visualization platform by Airbnb
After=multi-user.target

[Service]
Type=simple
User=chaoran
ExecStart=/home/chaoran/anaconda3/envs/superset/bin/gunicorn -w 2 --timeout 60 -b  0.0.0.0:8088 --limit-request-line 0 --limit-request-field_size 0 superset:app

[Install]
WantedBy=default.target
```

## start systemd service

```
# start superset service
sudo systemctl start superset.service
```

```
# check service status
sudo systemctl status superset.service
```
If everything is running ok, this is the expected output.

![systemd-status](https://raw.githubusercontent.com/6chaoran/data-story/master/data-tools/superset/systemd-service.png)

Now superset should be available at http://localhost:8088.


