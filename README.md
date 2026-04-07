# Black-Globe-Measurements

This repository is made to help share code, data and papers for our bachelorproject. In this project, we aim to research thermal-comfort indicators, and apply these indicators to moving subjects. Currently, the most fruitful indictors seem to be

- Mean Radiant Temperature (MRT)
- Wet-Bulb Globe Temperature (WBGT)

Our most recent theoretical investigation of MRT indicates that commonly used measuring methods are unjustified, this includes our mobile black globe thermometer setup.

## Mobile Measurements
The central assumption, when using black globe thermometers, in the popularly cited formula
$$
\boxed{T_{\text{MRT}} = \left[T_g^4 + \frac{h}{a_g\sigma}\left(T_g - T_a\right)\right]^{\frac{1}{4}}}
$$
Is that the thermometer is in thermal equilibrium. Disregarding the issue that this formula is fundamentally wrong (due to unphysical assumption), we still have the issue of never being in perfect equilibrium. For indoor measurements, were the use of globe thermometers originated, we can wait awhile and reasonably assume equilibrium. This is because the enviroment doesn't significantly change over the course of our measurement time. In outdoor settings, and in particular for mobile measurements, the enviroment (and therefore the MRT) isn't constant. We might imagine a moving subject moving from the shade to the sunny area.

For our bachelorproject, we therefore need to be able to recover the true MRT from globe thermometers that aren't in equilibrium. In our research, we found that linearizing the underlying ODE governing globe thermometer measurements yields a exponential smoothing behaviour. First degree ODEs of the form
$$
\tau \frac{\mathrm{d}y}{\mathrm{d}x} = g(x) - y
$$
From which we find $y(t) = (e^{-\frac{t-x}{\tau}} * g(x))(t)$, which is convolution. Conceptually, when applied to data, we can view a convolution as moving average with an infinite window size (= same length as data set), where the weights exponentially decay. If we construct this matrix $M$, than we find in our case (after linearizing) that
$$
T_{\text{MRT}, est} = M T_{\text{MRT}}
$$
In theory, we can easily recover the MRT by using the inverse of $M$
$$
T_{\text{MRT}} = M^{-1}T_{\text{MRT}, est} 
$$
This approach, however, has a catch. Applying this inverse matrix forms an ill-posed problem. We can intuitivally see that, if $M$ represents a data smoothing operation, that the inverse will amplify noise. For continuous, theoretical functions, this doesn't pose a problem. But real data is subject to statistical noise, which will indeed be amplified. We therefore need to first smooth out the data before performing this operation.

We therefore move our attention to the problem of smoothing $T_{\text{MRT}, est} $. We assert that $T_{\text{MRT}, est}$ should be of class $C^2$, and we assume that underneath our measurements a 'perfect' continuous $T_{\text{MRT}, est}$ exists. Our goal is then to approximate this 'perfect' $T_{\text{MRT}, est}$. As, in general, we don't know which parametric familie this function might be a part of, we look at non-parametric methods. A promising method is via a smoothing spline. These splines, when constructed of degree $k$, have the nice property that they are of class $C^{k-1}$. Literature indicates that the best smoothing spline is constructed using the GCV criterion. Scipy has an implementation of this method.

We therefore purpose the following recovery algorithm
1. Approximate the functions $h(t)$ and $T_{\text{MRT}, est}$ by  $\hat{h}(t)$ and $\hat{T}_{\text{MRT}, est}$ using a smoothing spline based on the GCV criterion
2. Construct the exponential smoothing matrix $M$ based on the globe used
3. Apply $M^{-1}$ to this approximation $\hat{T}_{\text{MRT}, est}$ to obtain $T_{\text{MRT}}$

The constant $\tau$ can be a function time depending on the convective heat transfer coefficient $h$. For forced convection, for example, wind speed is a necessary parameter (which might change with time). Below we demonstrated the utility of the method impose a synthetic true MRT which we aimed to recover

![Alt text](https://github.com/NoahVerhoeven/Black-Globe-Measurements/blob/recovery_algorithm/Theoretical%20Background%20Globe%20Thermometer/Figures/Inverse-Exponential-Smoothing-Algorithm-Decreasing.png)

![Alt text](https://github.com/NoahVerhoeven/Black-Globe-Measurements/blob/recovery_algorithm/Theoretical%20Background%20Globe%20Thermometer/Figures/Inverse-Exponential-Smoothing-Algorithm-Constant.png)

![Alt text](https://github.com/NoahVerhoeven/Black-Globe-Measurements/blob/recovery_algorithm/Theoretical%20Background%20Globe%20Thermometer/Figures/Inverse-Exponential-Smoothing-Algorithm-Realistic.png)

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux)
4. Install the mrt_tools module: `pip install -e ./Code`

## Issues
If you experience the following issue:
```bash
& : File C:\user_specific_path\.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system. For more information, see about_Execution_Pol
icies at https:/go.microsoft.com/fwlink/?LinkID=135170.
At line:1 char:3
+ & c:\user_specific_path ...
+   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : SecurityError: (:) [], PSSecurityException
    + FullyQualifiedErrorId : UnauthorizedAccess
```
You need to run

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
in powershell. This allows the virtual environment activate script to actually run, otherwise the virtual environment won't be active and necessary modules might not be present.