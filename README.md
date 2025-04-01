
![ddf.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/uzBpL2zr2SvxtSX4V2m9H.png)

# **Painting-126-DomainNet**

> **Painting-126-DomainNet** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify paintings into 126 domain categories using the **SiglipForImageClassification** architecture.

![- visual selection(1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/UQ1T3CLDEbmbErA9-pCwQ.png)

*Moment Matching for Multi-Source Domain Adaptation* : https://arxiv.org/pdf/1812.01754

*SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features* https://arxiv.org/pdf/2502.14786

```py
Classification Report:
                         precision    recall  f1-score   support

       aircraft_carrier     0.8065    0.4717    0.5952       106
            alarm_clock     1.0000    0.7612    0.8644        67
                    ant     0.8095    0.7234    0.7640       188
                  anvil     0.3205    0.2066    0.2513       121
              asparagus     0.8242    0.8827    0.8525       324
                    axe     0.4028    0.4857    0.4404       175
                 banana     0.8986    0.8986    0.8986       286
                 basket     0.7251    0.7492    0.7370       331
                bathtub     0.0000    0.0000    0.0000        35
                   bear     0.8704    0.8647    0.8675       303
                    bee     0.9478    0.9440    0.9459       250
                   bird     0.8031    0.8807    0.8401       176
             blackberry     0.0000    0.0000    0.0000        11
              blueberry     0.8258    0.8258    0.8258       132
              bottlecap     0.6487    0.8173    0.7233       427
               broccoli     0.8961    0.8625    0.8790        80
                    bus     0.6909    0.8444    0.7600        90
              butterfly     0.9313    0.9613    0.9460       310
                 cactus     0.9583    0.9388    0.9485        98
                   cake     0.5290    0.5984    0.5615       122
             calculator     0.0000    0.0000    0.0000        10
                  camel     0.8244    0.9351    0.8763       231
                 camera     0.9725    0.8480    0.9060       125
                 candle     0.7763    0.8551    0.8138       207
                 cannon     0.0000    0.0000    0.0000        43
                  canoe     0.8400    0.9161    0.8764       298
                 carrot     0.9744    0.9005    0.9360       211
                 castle     0.9027    0.9278    0.9151       180
                    cat     0.8824    0.9818    0.9294       275
            ceiling_fan     1.0000    0.1333    0.2353        30
             cell_phone     0.7117    0.7453    0.7281       106
                  cello     0.8647    0.9127    0.8880       126
                  chair     0.8750    0.1667    0.2800        42
             chandelier     0.9773    0.9348    0.9556        46
             coffee_cup     0.9015    0.8095    0.8530       147
                compass     0.9483    0.8871    0.9167        62
               computer     0.0000    0.0000    0.0000        14
                    cow     0.9590    0.9360    0.9474       125
                   crab     0.9829    0.9426    0.9623       122
              crocodile     0.9468    0.9271    0.9368        96
            cruise_ship     0.8977    0.8977    0.8977       176
                    dog     0.9149    0.9739    0.9435       574
                dolphin     0.8928    0.9595    0.9249       321
                 dragon     0.9278    0.9730    0.9499       185
                  drums     0.8457    0.8405    0.8431       163
                   duck     0.9335    0.9642    0.9486       335
               dumbbell     0.9539    0.9603    0.9571       151
               elephant     0.9405    0.9794    0.9595       339
             eyeglasses     0.5417    0.1970    0.2889        66
                feather     0.9314    0.9416    0.9365       274
                  fence     0.0000    0.0000    0.0000        39
                   fish     0.8829    0.9671    0.9231       304
               flamingo     0.9778    0.9888    0.9832       178
                 flower     0.7188    0.7706    0.7438       388
                   foot     0.5893    0.4853    0.5323        68
                   fork     0.9500    0.2836    0.4368        67
                   frog     0.9172    0.9925    0.9534       134
                giraffe     0.9762    0.9762    0.9762        84
                 goatee     0.4565    0.4828    0.4693        87
                 grapes     0.8761    0.8200    0.8471       250
                 guitar     0.8827    0.8827    0.8827       162
                 hammer     0.0000    0.0000    0.0000        36
             helicopter     0.9733    0.8835    0.9262       206
                 helmet     0.0000    0.0000    0.0000        22
                  horse     0.9514    0.9856    0.9682       417
               kangaroo     0.9387    0.9053    0.9217       169
                lantern     0.6263    0.7126    0.6667       174
                 laptop     0.8800    0.8871    0.8835       124
                   leaf     0.7754    0.8930    0.8301       402
                   lion     0.9347    0.8883    0.9109       403
               lipstick     0.9281    0.9045    0.9161       157
                lobster     0.9646    0.9455    0.9550       202
             microphone     0.9231    0.8136    0.8649       118
                 monkey     0.7892    0.8656    0.8256       320
               mosquito     0.8696    0.3846    0.5333        52
                  mouse     0.8610    0.9174    0.8883       351
                    mug     0.8669    0.9365    0.9003       299
               mushroom     0.9070    0.9653    0.9353       202
                  onion     0.8700    0.9231    0.8958       377
                  panda     0.9631    0.9952    0.9789       210
                 peanut     0.5000    0.1212    0.1951        66
                   pear     0.9278    0.9356    0.9317       357
                   peas     0.8281    0.7465    0.7852        71
                 pencil     0.4902    0.5245    0.5068       143
                penguin     0.9496    0.9576    0.9536       354
                    pig     0.9392    0.9500    0.9446       260
                 pillow     0.7273    0.0727    0.1322       110
              pineapple     0.9849    0.9812    0.9831       266
                 potato     1.0000    0.0652    0.1224        46
           power_outlet     0.9600    0.8889    0.9231        81
                  purse     0.5000    0.0513    0.0930        39
                 rabbit     0.8961    0.9673    0.9303       214
                raccoon     0.9490    0.9394    0.9442       198
             rhinoceros     0.9657    0.9657    0.9657       175
                  rifle     0.8200    0.8542    0.8367       192
              saxophone     0.8100    0.8556    0.8322       284
            screwdriver     0.7083    0.6296    0.6667        54
             sea_turtle     0.9757    0.9969    0.9862       322
                see_saw     0.3527    0.6077    0.4463       130
                  sheep     0.9328    0.9398    0.9363       266
                   shoe     0.9522    0.9567    0.9544       208
             skateboard     0.4464    0.2083    0.2841       120
                  snake     0.8627    0.8550    0.8588       338
              speedboat     0.8710    0.6835    0.7660        79
                 spider     0.8129    0.6975    0.7508       162
               squirrel     0.9325    0.9063    0.9192       427
             strawberry     0.9316    0.9470    0.9392       302
            streetlight     0.7493    0.7948    0.7714       346
            string_bean     0.8636    0.4130    0.5588        46
              submarine     0.5845    0.7423    0.6541       326
                   swan     0.9222    0.8910    0.9063       266
                  table     0.0000    0.0000    0.0000        81
                 teapot     0.8619    0.9318    0.8955       308
             teddy-bear     0.8517    0.9136    0.8816       220
             television     0.0000    0.0000    0.0000        40
       the_Eiffel_Tower     0.9366    0.9882    0.9617       254
the_Great_Wall_of_China     0.8244    0.8710    0.8471       124
                  tiger     0.9504    0.9702    0.9602       336
                    toe     0.0000    0.0000    0.0000         1
                  train     0.9367    0.9628    0.9496       323
                  truck     0.8864    0.7959    0.8387        98
               umbrella     0.6309    0.8174    0.7121       230
                   vase     0.7382    0.8309    0.7818       207
             watermelon     0.9479    0.9450    0.9464       327
                  whale     0.8877    0.8657    0.8766       283
                  zebra     0.9832    0.9832    0.9832       238

               accuracy                         0.8533     24032
              macro avg     0.7686    0.7273    0.7299     24032
           weighted avg     0.8445    0.8533    0.8424     24032
```


The model categorizes images into the following 126 classes:
- **Class 0:** "aircraft_carrier"
- **Class 1:** "alarm_clock"
- **Class 2:** "ant"
- **Class 3:** "anvil"
- **Class 4:** "asparagus"
- **Class 5:** "axe"
- **Class 6:** "banana"
- **Class 7:** "basket"
- **Class 8:** "bathtub"
- **Class 9:** "bear"
- **Class 10:** "bee"
- **Class 11:** "bird"
- **Class 12:** "blackberry"
- **Class 13:** "blueberry"
- **Class 14:** "bottlecap"
- **Class 15:** "broccoli"
- **Class 16:** "bus"
- **Class 17:** "butterfly"
- **Class 18:** "cactus"
- **Class 19:** "cake"
- **Class 20:** "calculator"
- **Class 21:** "camel"
- **Class 22:** "camera"
- **Class 23:** "candle"
- **Class 24:** "cannon"
- **Class 25:** "canoe"
- **Class 26:** "carrot"
- **Class 27:** "castle"
- **Class 28:** "cat"
- **Class 29:** "ceiling_fan"
- **Class 30:** "cell_phone"
- **Class 31:** "cello"
- **Class 32:** "chair"
- **Class 33:** "chandelier"
- **Class 34:** "coffee_cup"
- **Class 35:** "compass"
- **Class 36:** "computer"
- **Class 37:** "cow"
- **Class 38:** "crab"
- **Class 39:** "crocodile"
- **Class 40:** "cruise_ship"
- **Class 41:** "dog"
- **Class 42:** "dolphin"
- **Class 43:** "dragon"
- **Class 44:** "drums"
- **Class 45:** "duck"
- **Class 46:** "dumbbell"
- **Class 47:** "elephant"
- **Class 48:** "eyeglasses"
- **Class 49:** "feather"
- **Class 50:** "fence"
- **Class 51:** "fish"
- **Class 52:** "flamingo"
- **Class 53:** "flower"
- **Class 54:** "foot"
- **Class 55:** "fork"
- **Class 56:** "frog"
- **Class 57:** "giraffe"
- **Class 58:** "goatee"
- **Class 59:** "grapes"
- **Class 60:** "guitar"
- **Class 61:** "hammer"
- **Class 62:** "helicopter"
- **Class 63:** "helmet"
- **Class 64:** "horse"
- **Class 65:** "kangaroo"
- **Class 66:** "lantern"
- **Class 67:** "laptop"
- **Class 68:** "leaf"
- **Class 69:** "lion"
- **Class 70:** "lipstick"
- **Class 71:** "lobster"
- **Class 72:** "microphone"
- **Class 73:** "monkey"
- **Class 74:** "mosquito"
- **Class 75:** "mouse"
- **Class 76:** "mug"
- **Class 77:** "mushroom"
- **Class 78:** "onion"
- **Class 79:** "panda"
- **Class 80:** "peanut"
- **Class 81:** "pear"
- **Class 82:** "peas"
- **Class 83:** "pencil"
- **Class 84:** "penguin"
- **Class 85:** "pig"
- **Class 86:** "pillow"
- **Class 87:** "pineapple"
- **Class 88:** "potato"
- **Class 89:** "power_outlet"
- **Class 90:** "purse"
- **Class 91:** "rabbit"
- **Class 92:** "raccoon"
- **Class 93:** "rhinoceros"
- **Class 94:** "rifle"
- **Class 95:** "saxophone"
- **Class 96:** "screwdriver"
- **Class 97:** "sea_turtle"
- **Class 98:** "see_saw"
- **Class 99:** "sheep"
- **Class 100:** "shoe"
- **Class 101:** "skateboard"
- **Class 102:** "snake"
- **Class 103:** "speedboat"
- **Class 104:** "spider"
- **Class 105:** "squirrel"
- **Class 106:** "strawberry"
- **Class 107:** "streetlight"
- **Class 108:** "string_bean"
- **Class 109:** "submarine"
- **Class 110:** "swan"
- **Class 111:** "table"
- **Class 112:** "teapot"
- **Class 113:** "teddy-bear"
- **Class 114:** "television"
- **Class 115:** "the_Eiffel_Tower"
- **Class 116:** "the_Great_Wall_of_China"
- **Class 117:** "tiger"
- **Class 118:** "toe"
- **Class 119:** "train"
- **Class 120:** "truck"
- **Class 121:** "umbrella"
- **Class 122:** "vase"
- **Class 123:** "watermelon"
- **Class 124:** "whale"
- **Class 125:** "zebra"

# **Run with Transformers🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Painting-126-DomainNet"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def painting_classification(image):
    """Predicts the painting category for an input image."""
    # Convert the input numpy array to a PIL image and ensure it is in RGB format
    image = Image.fromarray(image).convert("RGB")
    
    # Process the image for the model
    inputs = processor(images=image, return_tensors="pt")
    
    # Get predictions from the model without gradient computation
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Convert logits to probabilities using softmax
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    # Define the label mapping for each class index
    labels = {
        "0": "aircraft_carrier", "1": "alarm_clock", "2": "ant", "3": "anvil", "4": "asparagus",
        "5": "axe", "6": "banana", "7": "basket", "8": "bathtub", "9": "bear",
        "10": "bee", "11": "bird", "12": "blackberry", "13": "blueberry", "14": "bottlecap",
        "15": "broccoli", "16": "bus", "17": "butterfly", "18": "cactus", "19": "cake",
        "20": "calculator", "21": "camel", "22": "camera", "23": "candle", "24": "cannon",
        "25": "canoe", "26": "carrot", "27": "castle", "28": "cat", "29": "ceiling_fan",
        "30": "cell_phone", "31": "cello", "32": "chair", "33": "chandelier", "34": "coffee_cup",
        "35": "compass", "36": "computer", "37": "cow", "38": "crab", "39": "crocodile",
        "40": "cruise_ship", "41": "dog", "42": "dolphin", "43": "dragon", "44": "drums",
        "45": "duck", "46": "dumbbell", "47": "elephant", "48": "eyeglasses", "49": "feather",
        "50": "fence", "51": "fish", "52": "flamingo", "53": "flower", "54": "foot",
        "55": "fork", "56": "frog", "57": "giraffe", "58": "goatee", "59": "grapes",
        "60": "guitar", "61": "hammer", "62": "helicopter", "63": "helmet", "64": "horse",
        "65": "kangaroo", "66": "lantern", "67": "laptop", "68": "leaf", "69": "lion",
        "70": "lipstick", "71": "lobster", "72": "microphone", "73": "monkey", "74": "mosquito",
        "75": "mouse", "76": "mug", "77": "mushroom", "78": "onion", "79": "panda",
        "80": "peanut", "81": "pear", "82": "peas", "83": "pencil", "84": "penguin",
        "85": "pig", "86": "pillow", "87": "pineapple", "88": "potato", "89": "power_outlet",
        "90": "purse", "91": "rabbit", "92": "raccoon", "93": "rhinoceros", "94": "rifle",
        "95": "saxophone", "96": "screwdriver", "97": "sea_turtle", "98": "see_saw", "99": "sheep",
        "100": "shoe", "101": "skateboard", "102": "snake", "103": "speedboat", "104": "spider",
        "105": "squirrel", "106": "strawberry", "107": "streetlight", "108": "string_bean",
        "109": "submarine", "110": "swan", "111": "table", "112": "teapot", "113": "teddy-bear",
        "114": "television", "115": "the_Eiffel_Tower", "116": "the_Great_Wall_of_China",
        "117": "tiger", "118": "toe", "119": "train", "120": "truck", "121": "umbrella",
        "122": "vase", "123": "watermelon", "124": "whale", "125": "zebra"
    }
    
    # Map each label to its corresponding probability (rounded)
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    return predictions

# Create Gradio interface for the painting classifier
iface = gr.Interface(
    fn=painting_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Painting-126-DomainNet Classification",
    description="Upload a painting to classify it into one of 126 categories."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```

# **Intended Use:**

The **Painting-126-DomainNet** model is designed for painting image classification. It categorizes paintings into a wide range of domains—from objects like an "aircraft_carrier" or "alarm_clock" to animals, plants, and everyday items. Potential use cases include:

- **Art Curation & Analysis:** Assisting galleries and museums in organizing and categorizing artworks.
- **Creative Search Engines:** Enabling painting-based search for art inspiration and research.
- **Educational Tools:** Supporting art education by categorizing and retrieving visual resources.
- **Computer Vision Research:** Providing a benchmark dataset for studies in painting recognition and domain adaptation tasks.
