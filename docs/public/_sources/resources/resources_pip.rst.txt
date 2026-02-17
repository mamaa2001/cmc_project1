.. _sec-resources:pip:

How to use pip3
===============


.. _sec-lin:venv-installpackages:

Installing packages with Pip3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When your virtual environment is active, you can easily install packages using :math:`pip`. 

.. code:: bash

   $ pip3 install <packagename>

Note that you can also specify the version of the package you want to install.

.. code:: bash

   $ pip3 install <packagename>==1.15.4

.. _sec-lin:venv-uninstallpackages:

Uninstalling packages with Pip3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uninstalling a package with :math:`pip` is as easy as running the following command.

.. code:: bash

   $ pip3 uninstall numpy

.. _sec-lin:venv-installedpackages:

Check installed packages
~~~~~~~~~~~~~~~~~~~~~~~~

To get a collection of the packages installed in the environment.

.. code:: bash

   $ pip3 list

Install required packages
~~~~~~~~~~~~~~~~~~~~~~~~

Now install the reuired packages via

.. code:: bash

   $ pip3 install -r requirements.txt

