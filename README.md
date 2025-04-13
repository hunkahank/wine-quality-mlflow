# wine-quality-mlflow
This is a tutorial for ML flow.

## Code Generation and Licensing

The initial version of the core training script (`src/train.py`) in this project was generated with assistance from Google's AI Assistant (Gemini) on April 13, 2025.

**License Information:**

* The code provided by the AI assistant is offered "as-is" without any warranty, express or implied.
* You, the user, are responsible for ensuring your use of the code complies with all applicable laws, regulations, and Google's AI terms of service.
* **Crucially**, this project relies on external libraries (like `pandas`, `scikit-learn`, `numpy`, `mlflow`). These libraries have their *own* open-source licenses (e.g., BSD, MIT, Apache 2.0). You **must** comply with the terms of these licenses when using, redistributing, or creating derivative works from this project. Please review the specific licenses for each dependency.
* Responsibility for the application, modification, redistribution, or creation of derivative works based on this code lies solely with the user.

## Viewing Experiments with MLflow UI

MLflow includes a web-based user interface (UI) that allows you to visualize, search, and compare your experiment runs, including logged parameters, metrics, and artifacts.

**Prerequisites:**

* You have run the training script (`make run`) at least once, which creates the `mlruns` directory containing the tracking data.
* You have MLflow installed in your environment (`make install`).

**Running the MLflow UI:**

The command to start the UI is `mlflow ui`. Where you run it and how you access it depends on whether you are running it locally or on a remote server.

**1. Local Machine Access**

If you are working directly on your local computer (e.g., your laptop):

1.  Open your terminal.
2.  Navigate to the root directory of this project (`wine-quality-mlflow/`), where the `mlruns` directory is located.
3.  Run the command:
    ```bash
    mlflow ui
    ```
4.  MLflow will start a server, typically listening on `http://127.0.0.1:5000`. Open this URL in your web browser.

**2. Remote Server Access (e.g., EC2, Cloud VM)**

If your code and the `mlruns` directory are on a remote Linux server (like an AWS EC2 instance) and you want to access the UI from your local machine's browser:

1.  Connect to your remote server via SSH.
2.  Navigate to the root directory of this project (`wine-quality-mlflow/`), where the `mlruns` directory is located.
3.  Run the command using `--host 0.0.0.0` to allow external connections:
    ```bash
    mlflow ui --host 0.0.0.0 --port 5000
    ```
    * `--host 0.0.0.0`: Makes the server listen on all available network interfaces, not just localhost. **This is essential for remote access.**
    * `--port 5000`: Specifies the port (5000 is the default, but it's good practice to be explicit).
4.  Keep the terminal window with the `mlflow ui` command running.

**Important Steps for Remote Access:**

* **Firewall / Security Group:** You **must** ensure that the server's firewall allows incoming TCP connections on the port you specified (e.g., 5000).
    * On AWS EC2, this means configuring the instance's **Security Group** to add an **Inbound Rule** for `Custom TCP`, Port `5000`, and setting the **Source** to your IP address (`My IP` in the AWS console is often easiest) or a specific IP range. Using `0.0.0.0/0` for the source will allow anyone, which is less secure and generally not recommended for production.
    * On other Linux systems, you might need to configure `ufw` or `firewalld`.
* **Accessing the UI:** Find the **Public IPv4 address** of your remote server (e.g., from the EC2 console). Open your local web browser and navigate to:
    ```
    http://<your-server-public-ip-address>:5000
    ```
    (Replace `<your-server-public-ip-address>` with the actual public IP).

**Stopping the UI Server:**

Go back to the terminal where `mlflow ui` is running and press `Ctrl+C`.