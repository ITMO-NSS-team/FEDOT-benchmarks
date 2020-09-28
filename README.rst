AutoML-benchmark
================

This tool will help you to execute different AutoMl frameworks with
problem data you want. The repository already has some cases
(i.e. credit_scoring), the ability to work with PMLB datasets and open
to new experiments.

Installation
------------
AutoML benchmark includes
`FEDOT framework <https://github.com/nccr-itmo/FEDOT>`__ as a submodule.

To work with the FEDOT submodule without extra efforts and mistakes
follow the steps:

1. To clone module with the content of submodule

   ::

   $ git clone –-recursive https://github.com/ITMO-NSS-team/AutoML-benchmark


2. From the project root directory create a soft link to the core of
   the FEDOT Framework typing following command in terminal:

   Linux/OSX

   ::

   $ ln -s FEDOT/core core



   Windows (run terminal as administrator)

   ::

   $ mklink /j "core" "FEDOT/core"


**Please, do not add your link directory to the commits**. If you don’t want
to use the link anymore type the following command from the project root
directory:

Linux/OSX

::

$ unlink core

Windows (run terminal as administrator)

::

$ rmdir core

