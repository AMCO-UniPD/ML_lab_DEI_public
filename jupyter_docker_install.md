# Install Docker + Jupyter Lab

Search on the web for: "install docker desktop" + "windows" or "mac os" depending on the computer you have.
- For Mac:
  - Link: [https://docs.docker.com/desktop/setup/install/mac-install/](https://docs.docker.com/desktop/setup/install/mac-install/)
  - If you have an Apple Silicon CPU, download the version with Apple Silicon in the link, otherwise download the Intel version (blue button).
- For Windows:
  - Link: [https://docs.docker.com/desktop/setup/install/windows-install/](https://docs.docker.com/desktop/setup/install/windows-install/)
  - If you have an Intel/AMD CPU, download x86_64, otherwise, if you have ARM, download ARM (blue button).

Windows
![Pasted image 20250228165346](https://github.com/user-attachments/assets/db8f59bd-8d86-4a3a-bd47-8007b768b3cc)

Mac OS:
![Pasted image 20250228165415](https://github.com/user-attachments/assets/e18501d7-1b1d-465e-a4b7-dcdac2449b89)

Run the installer (I’m doing it on Windows, for example)
![Pasted image 20250228165534](https://github.com/user-attachments/assets/8f08a98e-d92c-4413-9ef0-62c6233d4043)
➡ Click on "Ok"

You should see something like this:
![Pasted image 20250228165755](https://github.com/user-attachments/assets/6fb05676-0eae-4164-a921-c2149ee34087)

On my PC, it took a couple of minutes, it also depends on your internet connection speed.

We can close the installer.
![Pasted image 20250228165849](https://github.com/user-attachments/assets/dc8bac65-11cc-40d4-96b2-47a7a673f337)

Accept the terms.
![Pasted image 20250228165947](https://github.com/user-attachments/assets/017f8d9c-71f9-4a09-ba35-d77de81b4fb3)

You can click "skip" at the top right (you don’t need an account).
![Pasted image 20250228170100](https://github.com/user-attachments/assets/91795fe7-afde-47a6-a402-261c27390ce4)

Click on the search bar and search for "jupyter datascience".
![Pasted image 20250228170217](https://github.com/user-attachments/assets/7b688654-b1a3-42a6-bece-84ac2ac3fa71)

Click "pull" on the correct image, the one with the Jupyter icon called "jupyter/datascience-notebook".
![Pasted image 20250228170313](https://github.com/user-attachments/assets/c5e57837-cd75-42e8-a3f0-704f17a18781)

After a few minutes (again, depending on your internet connection speed), you should see the image among the available ones, so click on Run (▶ icon).
![Pasted image 20250228170642](https://github.com/user-attachments/assets/1cb470f9-744d-4510-aa72-e1f506dce0dc)

Click on "Optional settings" and write "8888" in the "Host port" field.
![Pasted image 20250228171230](https://github.com/user-attachments/assets/2bf41ecc-60a1-4f1f-8d90-a13289e38d37)
Then click on the "Run" button.

You will see output similar to this, click on the last link.
![Pasted image 20250228171357](https://github.com/user-attachments/assets/0c3b2c5e-51f4-424d-b935-b1db6641411b)

Jupyter Lab will open, just like we saw in class.
![Pasted image 20250228171437](https://github.com/user-attachments/assets/a2a3c9b0-496a-4578-9010-c461686ccd86)

Click on the terminal icon, and you can git clone our repository with the command:
```
git clone https://github.com/AMCO-UniPD/ML_lab_DEI_public.git
```
And press enter, you will se an output similar to this one:
![Pasted image 20250228171544](https://github.com/user-attachments/assets/5106537a-30c8-4cda-9c6e-b0b87f76e8f3)
And in the file pane, you will see the folder "ML_lab_DEI_public" (press the refresh button if you don't see it)

Double-click to open the folder, inside you will see the course notebooks, which you can always open by double-clicking.
![Pasted image 20250228171651](https://github.com/user-attachments/assets/c4fe2ee6-3456-4ef3-b8e8-ab1ba85f147c)
