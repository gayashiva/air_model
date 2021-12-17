<p align="center">
<a href="url"><img align="center" alt="screen recorder" src="https://github.com/Gayashiva/air_model/blob/dev/src/web/logos/AIR_logo_circle.png" align="left" height="148" width="148" ></a>
</a>
</p>
<p align="center">
<h1 align="center">Artificial Ice Reservoirs</h1>
</p>
<p align="center"> A physical model that estimates the meltwater quantity and survival duration of artificial ice reservoirs (aka Icestupas). </p>

<!-- --- -->

<!-- <h3 align="center"> -->
<!--    <a href="https://share.streamlit.io/gayashiva/air_model/src/web/app.py">Run model</a>  -->
<!-- </h3> -->

<!-- --- -->

## Timelapses

![](src/web/logos/Guttannen_Icestupa_2021.gif)

The full timelapses can be viewed here:

[Guttannen 2020-21](https://youtu.be/kXi4abO4YVM) 

[Guttannen 2019-20](https://youtu.be/kcrvhU20OOE) 

[Fribourg 2018-19](https://youtu.be/GhljRBGpxMg)

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module and provides logging setup
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── settings.py
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── icestupaClass.py
    │   │   └── userApp.py <- Scripts to generate model results
    │   │
    │   ├── logs           <- Scripts for logger configuration and output
    │   │
    │   └── visualization  <- Scripts to create streamlit web app
    │       └── webApp.py
    │
    ├── .streamlit         <- Web app theming
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
