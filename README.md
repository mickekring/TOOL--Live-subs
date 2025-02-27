# Någorlunda realtidsundertexter till video

## Vad?
Lite cowboy-kod som funkar lite som Proof of Concept för att generera undertext med KB Whisper, helt lokalt på din dator.  

## Hur funkar det?
När du kör scriptet så skapas en chroma key-grön bakgrund med storlek 1920 x 1080 pixlar. Över det 
läggs en svart ruta där den genererade texten printas.  
Om det är första gången du kör scriptet hämtas den modell av KB Whisper ned från Huggingface och sparas i mappen cache där du kör scriptet från. Det här kan ta en stund beroende på vilken modell du valt.  

Det går definitivt att snabba upp genereringen av undertexter, men det är ingen "hit" att bara översätta ord för ord. För att få med grammatik och liknande så har jag valt att åtminstone låta det gå ett tag innan översättning sker. Så det är ett par sekunders eftersläpning, men med bättre kvalitet.

**Keyboard Controls:**

ESC: Exit the application  
P: Pause/resume transcription  
H: Show/hide control overlay  
S: Save transcript to file

## Hur gör jag för att testa?
Jag ingen möjlighet att supporta detta, tyvärr. Jag kör det på en Macbook Pro M3 Max 64GB. Python 3.12.  

1. Ladda ned all kod
2. Skapa en virtual environment
    - För Python 3: `python3 -m venv venv`
    - För Python 2: `python -m venv venv`
4. Aktivera virtual environment
    - På Windows: `venv\Scripts\activate`
    - På windows med bash terminal: `source venv/Scripts/activate`
    - På macOS och Linux: `source venv/bin/activate`
5. När en virtuell miljö är aktiverad ändras vanligtvis din terminalprompt för att inkludera namnet på den virtuella miljön. Du kan kontrollera om miljövariabeln VIRTUAL_ENV är satt. Kör följande kommando i din terminal: `echo $VIRTUAL_ENV` Om den virtuella miljön är aktiverad kommer detta kommando att visa sökvägen till den virtuella miljön. Om den inte är aktiverad kommer det att returnera en tom rad.
6. Installera alla paket från `requirements.txt` med pip: `pip install -r requirements.txt`
7. Öppna app.py och leta rätt på 'FONT_PATH' och ändra till en path till ett typsnitt på din dator 
8. Spara och kör med `python app.py`

Sen finns det en massa variabler att skruva på.

### Möjliga problem

- Felmeddelande: RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work
  1. Ladda ner ffmpeg executable från https://ffmpeg.org/download.html
  2. Extrahera om det är en zip till valfri plats
  3. Lägg till adressen till mappen där ffmpeg exectuable finns i din dators Path environment variable
  4. Starta om din terminal
  5. Testa att `ffmpeg -version` ger resultat
  6. Testa att köra scriptet igen
- Felmeddelande: cv2.error: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window.cpp:1284: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvNamedWindow'
  1. Avinstallera opencv-python-headless: `pip uninstall opencv-python-headless`
  2. Installera opencv-python: `pip install opencv-python`
  3. Verifiera OpenCV-installation: `python cvtest.py`. Tryck på valfri tangent för att stänga testfönstret som skapas.
  4. Kör scriptet igen.
- Scriptet körs men jag ser inga undertexter.
  1. Tryck på H för att visa controlls. Då ska du se text i övra vänstra hörnet som t.ex. "Status: ready". (Om inte, kanske problemet är att den inte har hittat ett typsnitt?)
  2. Kolla om du ser en grön volymbar som rör på sig när du pratar. Om inte betyder det att programmet inte får in ljud. Dubbelkolla att du har rätt mikrofon selekterad som din default mikrofon på din dator, sen starta om scriptet.


## Version 
- v0.1.1 | Claude 3.7 löste en massa problem och skrev om koden. Bland annat så att långa meningar visas fullständigt.
- v0.02 | Första version som jag och O1 skrev.
