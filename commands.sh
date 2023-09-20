ssh-keygen -t ed25519 -C "your_email@example.com"
git clone git@github.com:Lewington-pitsos/spotify-albums.git
pip install virtualenv
virtualenv venv 
source venv/bin/activate
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
