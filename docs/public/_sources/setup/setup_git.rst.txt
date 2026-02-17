
.. _sec-setup:git_cmc_clone:

Setup CMC repository
++++++++++++++++++++

This document outlines the basic steps to clone and stay up to date with
the CMC-2026 exercises using Git. It is not a comprehensive tutorial on
using Git. For more information, check `this section <#sec-resources:git>`__

Cloning
-------

1. Open the terminal/bash

   - Windows : Open Git Bash or a command prompt from the start menu

   - Mac-OS : Open Terminal using Spotlight search. You can also go to Applications -> Utilities -> Terminal

   - Open a terminal using your default application search.


2.  Navigate to the location where you want to install the CMC exercises
    folder. For example, if you want to navigate to your Desktop:

    - Windows :

      .. code:: bash

                $ cd C:/Users/(YOUR_USER_NAME)/Desktop

    - Mac-OS / Linux:

      .. code:: bash

                $ cd ~/Desktop

3.  Then execute the following command to clone the repository:

    .. code:: bash

              $ git clone https://gitlab.com/farmsim/courses/cmc-2026-students.git

   This would clone the repository with the default folder name
   ``cmc-2026-students``. If you wish to clone with a different folder name, then use
   the following command and replace ``FOLDER_NAME`` with the name of
   the folder you want to clone into:

   .. code:: bash

             $ git clone https://gitlab.com/farmsim/courses/cmc-2026-students.git FOLDER_NAME
