# Naturalistic Neural Manifolds
Code for running the analyses in "Neural Manifolds of Human Intracranial Recordings During Naturalistic Arm Movements" by Z. Steine-Hanson, R. Rao and B. Brunton

This paper and code explore neural manifolds in naturalistic ECoG data 
Methods heavily influenced by Natraj et al. 2022 

To use this repo:
<ol>
    <li> Set up your environment:
        <ul>
            <li>Create a matching virtual environment using <a href="https://github.com/zoeis52/neural_manifolds/blob/main/ECoGDL_venv.yml"> ECoGDL_venv.yml </a> Note: may not be up to date for some packages. </li>
            <li>Then, in the virtual environment, pip install the src folder using: <br>
            <code> pip install -e . </code> <br>
            While in the home directory for the project code. <br>
            You can test that this worked by importing src in a python interpretter.
            <li> You will also need to download these other git repos and pip install them: </li>
                <ul>
                    <li> For loading in mat files: <a href="https://github.com/skjerns/mat7.3"> https://github.com/skjerns/mat7.3 </a> </li>
                </ul>
            </li>
        </ul>
    </li>
    <li> Create a new experiment parameter json file, and update the parameters you'd like to change. An example can be found in <a href="https://github.com/BruntonUWBio/NaturalisticNeuralManifolds">experiment_params</a>. <br>
    I like to name the files such that: 
        <ul>
        <li>the number indicates which dataset is used</li>
        <li> the alphabetic character indicates an over-arching experiment type, such as varying frequency bands for filtering </li>
        <li> the roman numeral indicates variations for that experiment </li>
        <li> For example: exp_params_1a_i.json uses the 4-class naturalistic movement data, with motor roi projection, for the first test </li>
        </ul>
    </li>
    <li> Decide if you want to run PCA on each movement separetly, or together. Most likely you will want to run them separately. </li>
