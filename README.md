# The-Drone-Rangers

### Testing
There is a suite of tests written in pytest to evaluate the end-to-end behavior of the backend. 

## Backend

First, it is recommended to create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then, run `pip install -r requirements.txt` to install the necessary python packages. The backend can be executed via `python ./server/main.py`.

A variety of tests for the backend are available, which can be executed by

1. `pytest tests/unit`
1. `pytest tests/integration`
1. `pytest tests/server`
1. `pytest tests/e2e`

## Frontend

There are two frontend applications, the Live Farm App and the Simulation App. Both depend on the same backend server and will not run correctly if it has not already been started.

### Live Farm App
To run the live farm management app, execute:

```
cd ./frontend/livefarm-app
npm install
npm run dev
```

The app will be available at `http://localhost:5174/` when running.

### Simulation App
To run the simulation app, execute:

```
cd ./frontend/simulation-app
npm install
npm run dev
```

The app will be available at `http://localhost:5173/` when running.
