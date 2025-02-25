# live-subs

## Vad?
Lite cowboy-kod som funkar lite som Proof of Concept för att generera undertext med KB Whisper, helt lokalt på din dator.  

## Hur funkar det?
När du kör scriptet så skapas en chroma key-grön bakgrund med storlek 1920 x 1080 pixlar. Över det 
läggs en svart ruta där den genererade texten printas.  
Om det är första gången du kör scriptet hämtas den modell av KB Whisper ned från Huggingface och sparas i mappen cache där du kör scriptet från. Det här kan ta en stund beroende på vilken modell du valt.  

**Keyboard Controls:**

ESC: Exit the application  
P: Pause/resume transcription  
H: Show/hide control overlay  
S: Save transcript to file

## Hur gör jag för att testa?
Jag ingen möjlighet att supporta detta, tyvärr. Jag kör det på en Macbook Pro M3 Max 64GB. Python 3.12.  

1. Ladda ned all kod
2. Öppna requirements.txt och installera alla paket med pip
3. Öppna app.py och leta rätt på 'FONT_PATH' och ändra till en path till ett typsnitt på din dator 
4. Spara och kör med 'python app.py'

Sen finns det en massa variabler att skruva på.

## Version 
- v0.1.1 | Claude 3.7 löste en massa problem och skrev om koden. Bland annat så att långa meningar visas fullständigt.
- v0.02 | Första version som jag och O1 skrev.