# PaLM2 POC App with LangChain

## Setup

1. If you don’t have Python installed, [install it from here](https://www.python.org/downloads/).
2. Clone this repository.

3. Navigate into the project directory:

   ```bash
   $ cd palm2-poc-python
   ```

4. Create a new virtual environment:

   ```bash
   $ python -m venv venv
   $ . venv/bin/activate
   ```

5. Install the requirements:

   ```bash
   $ pip install -r requirements.txt
   ```

6. Run the Contact Center Automation App:

   ```bash
   $ cd contact-centre
   $ streamlit run palm2-contact-centre-automation.py
   ```
7. Run the Trader Dashboard App:

   ```bash
   $ cd trader-dashboard
   $ streamlit run palm2-trader-dashboard.py
   ```