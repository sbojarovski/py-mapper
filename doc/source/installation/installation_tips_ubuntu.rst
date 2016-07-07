Installation tips: Ubuntu
=========================

The following steps set everything up in Ubuntu (tested with Ubuntu 15.04 “Vivid Vervet”):

*   If you have not done so already, go to “System Settings” → “Software and Update” and active the item “Community-maintained free and open-source software (universe)”.

*   Open a terminal and enter the following commands:

    .. code-block :: bash

        $ sudo apt-get install python-numpy python-scipy python-matplotlib python-wxtools \
        >   python-opengl python-pip graphviz libboost-all-dev
        $ pip install fastcluster --user
        $ pip install mapper --user
        $ pip install cmappertools --user

*   Add the directory ``$HOME/.local/bin`` to the search path so that the Mapper GUI can be started more conventiently.

    To do so, open the file ``.bashrc`` in your home directory, eg. by

    .. code-block :: bash

        $ gedit ~/.bashrc

    and insert the following lines at the end:

    .. code-block:: bash

        # set PATH so that it includes user's private bin if it exists
        if [ -d "$HOME/.local/bin" ] ; then
            PATH="${PATH+$PATH:}$HOME/.local/bin"
        fi

    Now save the file and type

    .. code-block:: bash

        $ source ~/.bashrc

    in the terminal window to read the modified file in.

*   Try Mapper out:

    .. code-block:: bash

        $ MapperGUI.py
