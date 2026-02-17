.. _sec:venv:

Setup Virtual Environment
+++++++++++++++++++++++++

Before installing the libraries that will be needed for the lectures, we will set up a virtual environment.
At its core, the main purpose of Python virtual environments is to create an isolated collection of packages for Python projects. 
This means that each project can have its own dependencies, regardless of those used by every other project.
Using a virtual environment will allow you to work separately on course-related exercises without the risk of conflicts with
other projects.

We will provide a basic guide for setting up your Python virtual environment, for additional information, check

.. _sec-resources:venv-installation:

Virtual environment installation
--------------------------------

.. note::

   While there are alternatives to :math:`venv`, like :math:`virtualenv` or :math:`anaconda`, we recommend using :math:`venv`
   as it is the standard to create virtual environments with Python 3.
   If you are using Python 3.3 or newer, the :math:`venv` is included in the Python standard library and requires no additional installation. 

.. _sec-resources:venv-setup:

Working with virtualenv
-----------------------

.. _sec-resources:venv-setup-creation:
 
Venv creation
~~~~~~~~~~~~~
To create a virtual environment, go to your project’s directory and run the following command:

.. code:: bash

   $ python3 -m venv my_virtual_environment

**NOTE** *my_virtual_environment* is just a placeholder for the name you wish to give to the environment, shorter names are preferable.

This will create a folder named my_virtual_environment containing the virtual Python installation.
Note that, in principle, the virtual environment can be created anywhere in your system's directories, it is however good practice
to store the environment together with the project(s) exploiting it.

.. note::
   You should exclude your virtual environment directory from your version control system using .gitignore or similar.

.. _sec-resources:venv-setup-activation:

Venv activation
~~~~~~~~~~~~~~~
Before you can start installing or using packages in your virtual environment you’ll need to activate it. 
Activating a virtual environment will put the virtual environment-specific python and :math:`pip` executables into your shell’s PATH.

On Windows:

.. code:: bash

   $ cd path/to/project/folder
   $ ./my_virtual_environment/Scripts/activate

On Mac-OS/Linux:

.. code:: bash

   $ cd path/to/project/folder
   $ source my_virtual_environment/bin/activate

You can confirm you’re in the virtual environment by checking the location of your Python interpreter.

On Windows:

.. code:: bash

   $ where python

On Mac-OS/Linux:

.. code:: bash

   $ which python

The first returned element should be in the my_virtual_environment directory:

.. code:: bash

   $ path/to/project/folder/my_virtual_environment/bin/python

As long as your virtual environment is activated, :math:`pip` will install packages into that specific environment and you’ll be able to 
import and use packages in your Python application.

.. note::
   If you are using an IDE like Visual Studio Code, make sure that the Python interpreter path corresponds to the
   one of your virtual environment.

Venv deactivation
~~~~~~~~~~~~~~~~~
If you want to switch projects or otherwise leave your virtual environment, simply run:

.. code:: bash

   $ deactivate

If you want to re-enter the virtual environment just follow the same instructions above about activating a virtual environment. 
There’s no need to re-create the virtual environment.


