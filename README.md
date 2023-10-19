# biometric_auth
Authentication Biometric system  based on facial recognition
 What is facial recognition? In essence, it's the process of identifying a person or object based on their appearance. When you observe an apple, your mind immediately labels it as such—a fundamental form of recognition. Face recognition is an extension of this principle. When you look at a friend or their photograph, you instinctively focus on their face. This deliberate act allows you to recognize them by their facial features.

The question then arises: how does facial recognition work? The process is straightforward. When you meet someone for the first time, you don't recognize them immediately. You carefully study their facial features and learn their identity. For instance, they might introduce themselves as Paulo. Your mind associates this name with the facial data gathered. Subsequently, you can recognize Paulo by their face.

Now, you might be wondering about coding facial recognition. The steps mirror real-life recognition:

    Training Data Gathering: Collect facial images of individuals to be recognized.
    Recognizer Training: Train the recognition system with the facial data and corresponding names.
    Recognition: Test the system with new facial images to see if it identifies them.

In order to get this informations The Local Binary Patterns Histograms (LBPH) Face Recognizer is used. Face detection  is realized utilizing local binary patterns histograms. This explanation provides a succinct overview of its functioning.
The LBPH face recognizer was developed as an enhancement to mitigate susceptible to variations in lighting conditions, which are inherent to real-life scenarios.

The underlying concept involves examining the local characteristics of an image, rather than considering the image as a whole. The LBPH algorithm achieves this by analyzing the local structure of an image, achieved through pixel-by-pixel comparisons with neighboring pixels.

To implement this approach, a 3x3 window is moved across the image. At each position (each local segment of the image), the pixel at the center is compared with its neighboring pixels. Pixels with intensity values less than or equal to the center pixel are denoted as 1, while others are denoted as 0. These 0/1 values under the 3x3 window are read in a clockwise manner, forming a binary pattern such as 11100011. This pattern corresponds to a specific area of the image, and this process is repeated across the entire image, resulting in a list of local binary patterns.

<img width="625" alt="lbp-labeling" src="https://github.com/Eleonora99/biometric_auth/assets/68509977/de1c3203-6b4b-42b3-a12d-bb96668159e4">

After training data it is possible to authenticate and profile user according their identity.
