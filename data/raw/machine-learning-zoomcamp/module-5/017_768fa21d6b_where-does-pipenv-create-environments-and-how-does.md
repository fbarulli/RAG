---
id: 768fa21d6b
question: Where does pipenv create environments and how does it name them?
sort_order: 17
---

Pipenv creates environments in different locations depending on the operating system:

- **OSX/Linux:** `~/.local/share/virtualenvs/folder-name_cryptic-hash`
- **Windows:** `C:\Users\<USERNAME>\.virtualenvs\folder-name_cryptic-hash`

For example:

- `C:\Users\Ella\.virtualenvs\code-qsdUdabf` (for module-05 lesson)

The environment name is based on the name of the last folder in the directory where the `pipenv install` command was executed. For example, if you run any pipenv command in the directory `~/home/user/Churn-Flask-app`, it will create an environment named `Churn-Flask-app-some_random_characters`, and its path will look like:

- `/home/user/.local/share/virtualenvs/churn-flask-app-i_mzGMjX`

All libraries for this environment will be installed inside this folder. To activate the environment, navigate back to the project folder and type `pipenv shell`. Essentially, the location of the project folder acts as an identifier for an environment, replacing any specific name.