# Information
OCR model based on these archival sources:
* ÖStA/AVA/UuK/NK/IsrK/0029 (https://www.archivinformationssystem.at/detail.aspx?ID=167442)

# Commands
ketos compile --workers 3 --random-split 0.8 0.1 0.1 -f page -o ava_v1.arrow *.xml
ketos train --workers 3 --output ava -f binary ava_v1.arrow

# Training
