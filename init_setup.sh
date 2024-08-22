


echo [$(date)] : "Starting init_setup.sh"


#create a virtual environment

echo [$(date)] : "Creating virtual environment"

conda create --prefix ./env python=3.11 -y

echo [$(date)] : "Virtual environment created"

# Activate the virtual environment

echo [$(date)] : "Activating virtual environment"

#activate the virtual environment and return the status code of the activation


source activate ./env 

echo [$(date)] : "Virtual environment activated "

conda init && conda activate ./env &&  echo "Terminal activated" || echo "Terminal activation failed"

echo [$(date)] : "Virtual environment activated in terminal successfully"





# Install the required packages

echo [$(date)] : "Installing required packages"

pip install -r requirements_dev.txt

echo [$(date)] : "Packages installed"

echo [$(date)] : "END"


