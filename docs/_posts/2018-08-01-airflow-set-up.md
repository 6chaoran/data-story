---
title: "It's time to upgrade your scheduler to Airflow"
date: 2018-08-01 15:11:01 +0800
categories: 
  - data-engineering
toc: true
toc_sticky: true
---
![logo](https://cdn-images-1.medium.com/max/2000/1*fSL_bB5OrIrsxnP6GdYR5Q.jpeg)

Airflow is an open source scheduling tool, incubated by Airbnb. Airflow is now getting popular and more Tech companies start using it.
Compared with our company's existing scheduling tool - crontab, it provides advantageous features, such as user-friendly web UI, multi-process/distributed executions,notification when failure/re-try. 
In this post, I'm going to record down my journey of airflow setup.

## Content
* 1.Install Airflow
* 2.Configure Airflow
* 3.Choices of Executors
* 4.Final Notes

## 1. Install Airflow

First of all, make sure python2.7 and pip are installed and upgraded to the latest.
create a directory for Airflow: `mkdir ~/airflow`
set it as the home directory: `export AIRFLOW_HOME='~/airflow'`

refer to [Airflow official website](https://airflow.apache.org/installation.html), install the current latest version, using:
```
pip install apache-airflow==1.9.0
```

once installation is completed, type `airflow version` to verify.
![airflow-version](https://github.com/6chaoran/data-story/raw/master/data-tools/airflow/image/airflow-version.png)

## 2.Configure Airflow

### 2.1.initialize the database

initialize the default database using following, and a database `airflow.db` file will be created.

```
airflow initdb
```

![airflow-initdb](https://github.com/6chaoran/data-story/raw/master/data-tools/airflow/image/airflow-initdb.png)

### 2.2. start your webUI/scheduler

run `airflow scheduler` to start airflow scheduler.
As the process is running fore-ground, open another terminal, and run `airflow webserver` to start your webUI.

if you also encountered error:

`OSError: [Errno 2] No such file or directory`, just make sure the python path is added correctly.

* use `which python` to check your python installation path.
* use `vim ~/.bashrc` to add the python path, e.g. `export PATH=$PATH:xxx`

reference:[https://blog.csdn.net/aubdiy/article/details/73930865](https://blog.csdn.net/aubdiy/article/details/73930865)

If everything is successful, you will be able see the Airflow Web UI (http://localhost:8080) as follow:
![airflow-webserver](https://github.com/6chaoran/data-story/raw/master/data-tools/airflow/image/airflow-webui.png)

__tips:__ if you are running airflow at remote server, you need set up port forwarding at your client side.(e.g. Windows: Putty > Connection > SSH > Auth > Tunnels > Add new forwarded port, Mac: ssh user@server.ip -L 8080:localhost:8080)

### 2.3. test your DAG
There are some sample DAGs pre-defined in airflow. 

* `airflow list_dags`, `airflow list_tasks` are useful commands to check the existing DAGs
* `airflow test`, `airflow run` and `airflow backfill` are useful commands to test your tasks.

Let's do some tests on the `tutorial` DAG:

#### a. list all regiestered DAGs:

```
airflow list_dags
``` 
![airflow-webserver](https://github.com/6chaoran/data-story/raw/master/data-tools/airflow/image/airflow-listdags.png)

#### b. list all tasks under `tutorial` DAG in tree strcture:

```
airflow list_task tutorial --tree
```
```
[2018-07-29 17:06:34,887] {__init__.py:45} INFO - Using executor SequentialExecutor
[2018-07-29 17:06:34,919] {models.py:189} INFO - Filling up the DagBag from /home/chaoran/airflow/image/dags
<Task(BashOperator): sleep>
    <Task(BashOperator): print_date>
<Task(BashOperator): templated>
    <Task(BashOperator): print_date>
```
#### c. you can test run `print_date` task in `tutorial` DAG, using `airflow test tutorial print date 2018-08-01`. The date can be arbitary.

run backfill from `2018-07-11` to `2018-07-12`:
```
airflow backfill tutorial -s 2018-07-11 -e 2018-07-12
```
you will notice the backfill job is registered in webUI as well:
![airflow-webui2](https://github.com/6chaoran/data-story/raw/master/data-tools/airflow/image/airflow-webui2.png)

## 3. Choices of Executors


Airflow offer different execution mode for different scenarios:

|Mode of Executor | Detail |
|-----------------|:-------|
|Sequential|single machine, single-process. Airflow out-of-box option.|
|Local|single machine, multi-process. need to use postgres database instead of sqlite|
|Celery|could be distributed in multi-machine. need to start workers to pick up tasks|

As I only have a single EC2 instance instead of airflow cluster, Local Executor mode will be the most appropreated.

### 3.1. Install Postregres Database

install postgres: `sudo apt-get install postgresql postgresql-contrib`


### 3.2. Setup Postegres Database

sign in database with superuser postgres: `sudo -u postgres psql`

```
CREATE ROLE ubuntu;
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES on database airflow to ubuntu;
ALTER ROLE ubuntu SUPERUSER;
ALTER ROLE ubuntu CREATEDB;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ubuntu;
ALTER ROLE ubuntu LOGIN;
```
__tips__: don't forget `;` when key the above command.
![airflow-postgre-db](https://github.com/6chaoran/data-story/raw/master/data-tools/airflow/image/airflow-postgres-db.png)

use `\du` to confirm the role ubuntu is properly set:

```
                                   List of roles
 Role name |                         Attributes                         | Member of 
-----------+------------------------------------------------------------+-----------
 postgres  | Superuser, Create role, Create DB, Replication, Bypass RLS | {}
 ubuntu    | Superuser, Create DB                                       | {}
```

use `\q` to quit the postgres mode

change the listen port: `/etc/postgresql/9.*/main/pg_hba.conf`

* ipv4 address to `0.0.0.0/0`
* ipv4 connection `md5` to `trust`

change configure the `postgresql.conf` file to open the listen address to all ip addresses:

* listen_addresses = '*'.

start a postgresql service

* `sudo service postgresql start`

### 3.3. Configue Airflow.cfg file

use `vim ~/airflow/airflow.cfg` to modify config file:

* change the mode of executor: `executor = LocalExecutor`
* change database connection: `sql_alchemy_conn = postgresql+psycopg2://ubuntu@localhost:5432/airflow`

Finally, re-initialize the airflow database `airflow initdb`. You will find `INFO - Using executor LocalExecutor`, meaning LocalExecutor is successful set up.

### 3.4 Restart Aiflow scheduler/UI

```
airflow scheduler -D
airflow webserver -D
```
__tips__: use `-D` flag to run daemonized process, which will alow program to run at background.   
Now my airflow setup is completed, I just need write my own DAG file and drop into `~/airflow/dags`

## 4. Final Notes

### 4.1 Additional Tips

1) run airflow process with -D flag so that the process will be daemonize, which means will run in background. for example:
```
airflow scheduler -D
airflow webserver -p 8080 -D
```

2) airflow webserver will have multiple (4 by default) workers, killing webserver workers process if you want to restart/shutdown your airflow. for example:
```
ps -ef | grep airflow
sudo kill -9 airflow-pids
```

3) can't daemonize the process
try delete the `.err` and `.pid` file in airflow folder. for example:
```
cd ~/airflow
rm airflow-webserver.err airflow-webserver.pid
```

4) DAGs triggered but never run or in queue
* make sure airflow scheduler is running
* make sure airflow workers are running
* make sure the pause toggle on the left side is turned on

### 4.2 DAG code submission
Due to some security concern, the DAG schudeling code is centralized and managed by Data Engineering team. 
DAG code is usually submitted to git and synchronized to airflow.
I simply create a crontab job to sync DAG repository from bitbucket to airflow DAG folder every miniute.


## Reference:
[Airflow official website](https://airflow.apache.org/installation.html)    
[Installing Apache Airflow on Ubuntu/AWS – A.R.G.O. – Medium](https://medium.com/a-r-g-o/installing-apache-airflow-on-ubuntu-aws-6ebac15db211)
