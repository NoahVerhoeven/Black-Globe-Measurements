# Black-Globe-Measurements

This repository is made to help share code, data and papers for our bachelorproject. In this project, we aim to research thermal-comfort indicators, and apply these indicators to moving subjects. Currently, the most fruitful indictors seem to be

- Mean Radiant Temperature (MRT)
- Wet-Bulb Globe Temperature (WBGT)

Our most recent theoretical investigation of MRT indicates that commonly used measuring methods are unjustified, this includes our mobile black globe thermometer setup.

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