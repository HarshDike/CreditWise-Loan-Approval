# CreditWise Loan Predictor

The React frontend calls a FastAPI backend, which loads `credit_wise_model.pkl` and runs the saved preprocessing and logistic-regression model.

## Run locally

1. Create and activate a Python virtual environment, then install the backend:
   `pip install -r backend/requirements.txt`
2. Start the API from the project root:
   `uvicorn backend.app:app --reload --port 8000`
3. In another terminal, install and start the frontend:
   `cd frontend`, then `npm install` and `npm run dev`
4. Open the URL printed by Vite (normally `http://localhost:8080`).

Use Node.js 22.12+ if possible. The current Vite/TanStack stack can warn or misbehave on older Node 20 releases.

For a deployed API, set `VITE_API_URL` before building the frontend.
