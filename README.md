# Elevfremmøde Automaten

Automaten beskrives omtrent således:
>   En applikation som kan genkende elevers ansigter, i en sammenhæng hvor det fungerer lige som fremmøde registrering.  
Når en elev er tilmeldt, forsøger systemet at genkende ham næste gang han kommer. Alle elevernes navne vises på en liste, men systemet spørger om han er den der gættes på. Eleven kan enten bekræfte dette, eller vælge det rigtige navn fra listen. Hvis eleven ikke er på listen, kan han tilføje sig selv. På den måde kan systemet også fungere som gæstebog.
Hver gang en elev genkendes eller registreres, optager systemet en bunke billeder af ham, som gemmes og indlæreres.  
Da vi ikke vil på kant med registerlovgivningen (og almindelig tillid fra eleverne), nøjes vi med at simulere elevfremmøde registrering. Der optages altså ingen log over hvornår eleven genkendes, og optagede billedfiler får ændret deres dato til 1. Januar 1970.

Følgende teknologier er involveret:
* Python
* OpenCV
* Gui (ja, men hvilket framework?)
* Git