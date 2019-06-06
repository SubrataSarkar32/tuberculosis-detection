# tuberculosis-detection

Here is a sample prediction run:

![picture](https://www.kaggleusercontent.com/kf/15272869/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..4R8LpY9CCy2mQZ-JFwms9g.Mva0w43IohVN6ETmD5B2dwpBwz0A4U4yy7oA2MooUBmtkqqs8I4zVSKpejuYYWO4dwFZ_s-rcVxIpknGMMOUmIcfruWyaAnhupC6LAQcjyLMTRkPxiIquXIMVl81S_--J2BINunpVoQyDGdCBItwW1LZ3T-wrEuLf11hx-oBgZApUoBm9S_J4ph-3xv4Y97V.MzHxtnP_lQkSb0MDDMxDfQ/__results___files/__results___22_0.png)
actual: ['PTB']
predicted: ['PTB']

![picture](https://www.kaggleusercontent.com/kf/15272869/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..4R8LpY9CCy2mQZ-JFwms9g.Mva0w43IohVN6ETmD5B2dwpBwz0A4U4yy7oA2MooUBmtkqqs8I4zVSKpejuYYWO4dwFZ_s-rcVxIpknGMMOUmIcfruWyaAnhupC6LAQcjyLMTRkPxiIquXIMVl81S_--J2BINunpVoQyDGdCBItwW1LZ3T-wrEuLf11hx-oBgZApUoBm9S_J4ph-3xv4Y97V.MzHxtnP_lQkSb0MDDMxDfQ/__results___files/__results___22_2.png)
actual: ['normal']
predicted: ['normal']

![picture](https://www.kaggleusercontent.com/kf/15272869/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..4R8LpY9CCy2mQZ-JFwms9g.Mva0w43IohVN6ETmD5B2dwpBwz0A4U4yy7oA2MooUBmtkqqs8I4zVSKpejuYYWO4dwFZ_s-rcVxIpknGMMOUmIcfruWyaAnhupC6LAQcjyLMTRkPxiIquXIMVl81S_--J2BINunpVoQyDGdCBItwW1LZ3T-wrEuLf11hx-oBgZApUoBm9S_J4ph-3xv4Y97V.MzHxtnP_lQkSb0MDDMxDfQ/__results___files/__results___22_4.png)
actual: ['normal']
predicted: ['normal']

![picture](https://www.kaggleusercontent.com/kf/15272869/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..4R8LpY9CCy2mQZ-JFwms9g.Mva0w43IohVN6ETmD5B2dwpBwz0A4U4yy7oA2MooUBmtkqqs8I4zVSKpejuYYWO4dwFZ_s-rcVxIpknGMMOUmIcfruWyaAnhupC6LAQcjyLMTRkPxiIquXIMVl81S_--J2BINunpVoQyDGdCBItwW1LZ3T-wrEuLf11hx-oBgZApUoBm9S_J4ph-3xv4Y97V.MzHxtnP_lQkSb0MDDMxDfQ/__results___files/__results___22_6.png)
actual: ['normal']
predicted: ['normal']



As we observe for a low number of samples for training this cnn model has produced rather acceptable results.

### Code for using provided models for prediction on your own local machine:

```python
import random,keras,cv2
import os

from keras.preprocessing import image
from keras.models import load_model

import numpy as np
from keras.models import model_from_json
import json

def give_prediction(filename=''):
            '''
            Call this function with a filename of an image to get predictions
            '''
            path = os.path.abspath(filename)
            json_file = open('model_3.json', 'r')

            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)

            # load weights into new model
            model.load_weights("model_3.h5")
            print("Loaded model from disk")
            
            img = cv2.imread(path)
            img = cv2.resize(img, (100, 100))
            pred = model.predict_classes(img.reshape(-1,100,100,3))
            class_label_list = ['cat','dog','human']
            print(class_label_list[pred[0]])
            return class_label_list[pred[0]]
```
This cnn is for multi class classifier. In the python notebook just add path to your dataset and load the dataset and it will included for training.

Link to the model:
* [JSON file for structure](https://www.kaggleusercontent.com/kf/15272869/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..afS-yIKHXbR17TQw8C2Xcg.uXt5b59MtI43LHYyynQ6x4ySQzz-uytzcqb6tl8kwIG-HGQlTrcEYm5G4b4a32PNag-Px0ZjC63l-SrFeX23KvtsOfXS0lv-hYH0Ek4KPRhp0-mrTr7EGT3P4rVYzX65hXnxXD1MOGUEw9PbJwA5mcWYtPhK4GexH3o1YAHaD-Ce8Q0pKs2J4FEVmhizvEmn.GsLLjqvpck6hnSZlfBDuEA/model_4.json)
* [HDF5 weight file](https://www.kaggleusercontent.com/kf/15272869/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..62CwKxV8pAxpUvpsAcH4yQ.OsFfSDKPw1Lc5cQyfwzJ-RXvMFgIAYAAYI_mIWLQ1nNvHEB-brIJz2rgVCAhHkQxn0pgcj-PlY6Mk0WXkC387eZhXhqdQaaYLP-s7Gm3lpyFT7GdbQ86Q0oMvF8n9nhZsCfIU8fAPwqwASCNzoHZPWOm-BDDnRsKov53mA1E-6FT_NVt0WMn8FOH8dXu87hE.cOlthE4W36oEAbxQ8Ds4OA/model_4.h5)

### Link to kaggle kernel

[using-cnn-detect-pulmonary-disease-in-xray](https://www.kaggle.com/subratasarkar32/using-cnn-detect-pulmonary-disease-in-xray/)

Hope you enjoy using this neural network!

If you feel you can improve this model free free to fork and submit a pull request.
