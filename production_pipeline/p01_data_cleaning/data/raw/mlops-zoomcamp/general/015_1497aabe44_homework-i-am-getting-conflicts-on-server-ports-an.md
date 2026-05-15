---
id: 1497aabe44
question: 'Homework: I am getting conflicts on server ports and cannot establish a
  connection to the MLflow server, why?'
sort_order: 15
---

Your port (5000) may be in use by some other process. To resolve this:

1. Run the following command to find out which process is using the port:

   ```bash
   lsof -i :5000
   ```
2. Either kill the process using that port or route to a different port. You can explicitly change the port with the following command:

   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
   ```