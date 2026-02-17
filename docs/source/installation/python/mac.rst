.. _sec:mac:

=========
 Mac-OSX
=========

.. _sec-mac:python:

Python
------

.. _sec-mac:checking-if-python:

Checking if python exists
~~~~~~~~~~~~~~~~~~~~~~~~~

Before installing please check if you already have python installed on
your computer. To do so open terminal application. Once terminal is open
execute the following commands,

.. code:: bash

   $ python -V

.. code:: bash

   $ python3 -V

If either of them return ``Python 3`` then you can skip the Python
installation section and continue with the rest.

.. _sec-mac:installation-python:

Python installation
~~~~~~~~~~~~~~~~~~~

To download and install Python use the link :
`MacOS <https://www.python.org/downloads/mac-osx/>`__

During installation step make sure you choose customize option like in
figure `4 <#fig:mac-py-step1>`__ and then confirm that all the check
boxes are selected like in figure `5 <#fig:mac-py-step2>`__

.. figure:: figures/python_install_4.png
   :alt: Python installation customization - Step 1
   :name: fig:mac-py-step1

   Python installation customization - Step 1

.. figure:: figures/python_install_5.png
   :alt: Python installation customization - Step 2
   :name: fig:mac-py-step2

   Python installation customization - Step 2

*Install Latest Version*

After installation to verify if everything is working open terminal
again and run the commands in section
`3.1.1 <#sec-mac:checking-if-python>`__.

.. _sec-mac:pip:

Pip
---

Python has a huge repository of packages that are widely used for
different functions. In order to obtain these packages there are several
package managers. The one we will be using during this course will be
the official package installer for Python called :math:`pip`.

.. _sec-mac:checking-if-pip:

Checking if pip exists
~~~~~~~~~~~~~~~~~~~~~~

If you installed Python based on the instructions above then :math:`pip` should
be installed by default. Or it may have been already installed on your
computer if Python had been pre-installed. To check if :math:`pip` exists, open
terminal and execute the following command:

:math:`pip` or :math:`pip3` depends on your system. Typically they
differentiate ones installed with python2 and python3 respectively.

.. code:: bash

   $ pip --version

.. code:: bash

   $ pip3 --version

If :math:`pip` is already installed then at least one of the above
commands should print the version of the :math:`pip` along with the python and
its version associated with it. **Make sure that the python version is 3**

.. _sec-mac:installation-pip:

Pip installation
~~~~~~~~~~~~~~~~

If you have verified that :math:`pip` is not installed on your computer then in
order to install :math:`pip` you are expected to have either cloned or
downloaded the exercise repository by now.

-  Open terminal application

-  Navigate to the location where you have downloaded the exercise
   repository. You can use the command :math:`cd` to change directories
   and :math:`pwd` to check you current directory.

-  Inside the exercise repository navigate to **extras** folder and
   execute the following command:

.. code:: bash

   $ python get-pip.py

*Make sure command *\ **python**\ *refers to python-3. To check use the
commands mentioned in Python Installation section to get the
corresponding python version. According use either python or python3
commands*

Check if you have installed everything correctly by referring to
`3.2.1 <#sec-mac:checking-if-pip>`__.

.. .. _sec-mac:spyder:

.. Spyder
.. ------

.. Python programs can be written and run in several ways, it can be simply
.. done on a terminal by running *python* or *ipython*. While this method
.. is limited for simple programs, larger programs will be written using a
.. text-editor or an Integrated Development Environment (IDE). Though it is
.. not necessary to have an IDE for programming in Python, having one will
.. bring many features that are useful while starting new

.. .. _sec-macinstallation-spyder:

.. Installation
.. ~~~~~~~~~~~~

.. -  Open terminal

.. -  Next, install spyder with the command:

..    .. code:: bash

..       $ pip install spyder

..    or

..    .. code:: bash

..       $ pip3 install spyder

.. .. _sec-mac:checking-if-spyder:

.. Checking spyder
.. ~~~~~~~~~~~~~~~

.. To check if spyder is installed, execute the following command from a
.. terminal.

.. .. code:: bash

..    $ spyder3

.. If everything is working then Spyder IDE should open and you are ready
.. to begin with the exercises.


Virtual Environment
-------------------

Before installing the libraries that will be needed for the lectures, we will set up a virtual environment.
At its core, the main purpose of Python virtual environments is to create an isolated collection of packages for Python projects. 
This means that each project can have its own dependencies, regardless of those used by every other project.
Using a virtual environment will allow you to work separately on course-related exercises without the risk of conflicts with
other projects.

We will provide a guide for setting up your Python virtual environment, for additional information, check the official :math:`PyPa` website 
`Installing packages using pip and virtual environments <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment/>`__

.. _sec-mac:venv-installation:

Virtual environment installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   If you are using Python 3.3 or newer, the :math:`venv` module is the preferred way to create and manage virtual environments. 
   :math:`venv` is included in the Python standard library and requires no additional installation. If you are using :math:`venv`, you may skip this section.

:math:`virtualenv` is an alternative to :math:`venv` that should be used for previous version of Python 3 or Python 2.
To install :math:`virtualenv`, open the terminal and run the following commands.

.. code:: bash

   $ python3 -m pip install virtualenv


.. _sec-mac:venv-creation:

Venv creation
~~~~~~~~~~~~~
To create a virtual environment, go to your project’s directory and run venv. If you are using :math:`virtualenv`, replace :math:`venv` with :math:`virtualenv` in the below commands.

.. code:: bash

   $ cd path/to/project/folder
   $ python3 -m venv my_virtual_environment

This will create a folder named my_virtual_environment containing the virtual Python installation.
Note that, in principle, the virtual environment can be created anywhere in your system's directories, it is however good practice
to store the environment together with the project(s) exploiting it.

.. note::
   You should exclude your virtual environment directory from your version control system using .gitignore or similar.


Venv activation
~~~~~~~~~~~~~~~
Before you can start installing or using packages in your virtual environment you’ll need to activate it. 
Activating a virtual environment will put the virtual environment-specific python and :math:`pip` executables into your shell’s PATH.

.. code:: bash

   $ cd path/to/project/folder
   $ source my_virtual_environment/bin/activate

You can confirm you’re in the virtual environment by checking the location of your Python interpreter:

.. code:: bash

   $ which python

It should be in the my_virtual_environment directory:

.. code:: bash

   $ path/to/project/folder/my_virtual_environment/bin/python

As long as your virtual environment is activated :math:`pip` will install packages into that specific environment and you’ll be able to 
import and use packages in your Python application.

.. note::
   If you are using an IDE like Visual Studio Code, make sure that the Python interpreter path corresponds to the
   one of your virtual environment.

.. _sec-mac:venv-deactivation:

Venv deactivation
~~~~~~~~~~~~~~~~~
If you want to switch projects or otherwise leave your virtual environment, simply run:

.. code:: bash

   $ deactivate

If you want to re-enter the virtual environment just follow the same instructions above about activating a virtual environment. 
There’s no need to re-create the virtual environment.

.. _sec-mac:venv-installpackages:

Installing packages
~~~~~~~~~~~~~~~~~~~
When your virtual environment is active, you can easily install packages using :math:`pip`. 

.. code:: bash

   $ pip install <packagename>

.. code:: bash

   $ pip3 install <packagename>

Note that you can also specify the version of the package you want to install.

.. code:: bash

   $ pip install <packagename>==1.15.4

.. code:: bash

   $ pip3 install <packagename>==1.15.4

Uninstalling packages
~~~~~~~~~~~~~~~~~~~~~
Uninstalling a package with :math:`pip` is as easy as running the following command.

.. code:: bash

   $ pip uninstall numpy

.. code:: bash

   $ pip3 uninstall numpy

Requirements
------------

The final step before starting of with the exercise is to install a few
necessary packages. We will be using :math:`pip` to this.

-  Open terminal (command prompt on Windows)

-  Navigate in the terminal to the exercise repository on your computer

-  Activate your virtual environment

-  Execute the following command once you are in the root of the
   repository:

   .. code:: bash

      $ pip install -r requirements.txt

   .. code:: bash

      $ pip3 install -r requirements.txt

   Use :math:`pip` or :math:`pip3` depending on the one that refers to
   python3

The *requirements.txt* installs the following packages:

-  numpy : Scientific computing package for python

-  matplotlib : Matlab like plotting tool for python

-  farms_pylog : Module for logging messages during code runtime

After successfully completing the installation steps in the previous
sections, you can now get started with programming Lab0. Python
is not just a computational tool but a very powerful programming
language. This means having to learn a few more extra concepts to get
your job done. There are a ton of references available online for those
who are interested in learning Python in depth. We will try to provide
the necessary references to help with the concepts that are useful
during the course as and when needed.
