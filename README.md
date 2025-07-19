# Quick guide of baseline_estimate
1. Download the code from github with command:

   	    git clone  https://github.com/ruijiuchen/baseline_estimate.git

2. Use the following command to install the dependencies listed in the "requirements.txt" file:

       pip install -r requirements.txt
   
This will install all the required packages specified in the requirements.txt file.

3. Install baseline_estimate program via:

   	   pip install .

4. Execute the code:
Before execute the code, make sure the paths in the file "pathsettings.toml" are correct.

       baseline_estimate --filename From20250705_05-20-00_08-00-00.root --histname spectrogram --l 10000  --ratio 1e-6 --outroot baseline_corrected.root --outhistname h2_baseline_removed


test# baseline_estimate
