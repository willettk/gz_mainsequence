
;+
; NAME:
;       
;	FIX_MERGER_DATA
;
; PURPOSE:
;
;	Convert the IDL SAV files from S. Kaviraj into reasonable FITS tables, separating mergers (N = 3003; Darg+10) from post-mergers (N = 370; Carpineti+11)
;
; INPUTS:
;
;
;
; OUTPUTS:
;
;
;
; KEYWORDS:
;
;
;
; EXAMPLE:
;
;	IDL> .r fix_merger_data
;
; NOTES:
;
;	
;
; REVISION HISTORY
;       Written by K. Willett                Jul 2014
;-

merger_path = '/Users/willettk/Astronomy/Research/GalaxyZoo/gzmainsequence/mergers'

; Load first set of merger data (SDSS metadata) from IDL sav files

restore,merger_path+'/merger_SDSS_data.sav'
openw,lun1,merger_path+'/merger_sdss_data.csv',/get_lun,width=1000

; Print column sizes

print,'PHOTO1         ',n_elements(PHOTO1)
print,'PHOTO2         ',n_elements(PHOTO2)
print,'ABSMAG1        ',n_elements(ABSMAG1)
print,'ABSMAG2        ',n_elements(ABSMAG2)
print,'PHOTO_ERR1     ',n_elements(PHOTO_ERR1)
print,'PHOTO_ERR2     ',n_elements(PHOTO_ERR2)
print,'SPEC_DATA1     ',n_elements(SPEC_DATA1)
print,'SPEC_DATA2     ',n_elements(SPEC_DATA2)
print,'SPECZ1         ',n_elements(SPECZ1)
print,'SPECZ2         ',n_elements(SPECZ2)
print,'PHOTOZ1        ',n_elements(PHOTOZ1)
print,'PHOTOZ2        ',n_elements(PHOTOZ2)
print,'RA1            ',n_elements(RA1)
print,'RA2            ',n_elements(RA2)
print,'DEC1           ',n_elements(DEC1)
print,'DEC2           ',n_elements(DEC2)
print,'VOTE1          ',n_elements(VOTE1)
print,'VOTE2          ',n_elements(VOTE2)
print,'SPEC_DATA_NAMES',n_elements(SPEC_DATA_NAMES)
print,'PHOTO_NAMES    ',n_elements(PHOTO_NAMES)
print,'KMASS1         ',n_elements(KMASS1)
print,'KMASS2         ',n_elements(KMASS2)
print,'KMASS_ERR1     ',n_elements(KMASS_ERR1)
print,'KMASS_ERR2     ',n_elements(KMASS_ERR2)

; Magnitudes are in gruiz order rather than ugriz; no idea why. 
; Re-order them to ugriz and separate each into a unique column

u_app_1 = photo1[2,*]
g_app_1 = photo1[0,*]
r_app_1 = photo1[1,*]
i_app_1 = photo1[3,*]
z_app_1 = photo1[4,*]

u_app_2 = photo2[2,*]
g_app_2 = photo2[0,*]
r_app_2 = photo2[1,*]
i_app_2 = photo2[3,*]
z_app_2 = photo2[4,*]

u_abs_1 = absmag1[2,*]
g_abs_1 = absmag1[0,*]
r_abs_1 = absmag1[1,*]
i_abs_1 = absmag1[3,*]
z_abs_1 = absmag1[4,*]

u_abs_2 = absmag2[2,*]
g_abs_2 = absmag2[0,*]
r_abs_2 = absmag2[1,*]
i_abs_2 = absmag2[3,*]
z_abs_2 = absmag2[4,*]

u_app_err_1 = photo_err1[2,*]
g_app_err_1 = photo_err1[0,*]
r_app_err_1 = photo_err1[1,*]
i_app_err_1 = photo_err1[3,*]
z_app_err_1 = photo_err1[4,*]

u_app_err_2 = photo_err2[2,*]
g_app_err_2 = photo_err2[0,*]
r_app_err_2 = photo_err2[1,*]
i_app_err_2 = photo_err2[3,*]
z_app_err_2 = photo_err2[4,*]

plate1   = spec_data1[0,*]
mjd1     = spec_data1[1,*]
fiberid1 = spec_data1[2,*]

plate2   = spec_data2[0,*]
mjd2     = spec_data2[1,*]
fiberid2 = spec_data2[2,*]

; Print header line to file

printf,lun1,'# u_app_1, g_app_1, r_app_1, i_app_1, z_app_1, u_app_2, g_app_2, r_app_2, i_app_2, z_app_2, u_app_err_1, g_app_err_1, r_app_err_1, i_app_err_1, z_app_err_1, u_app_err_2, g_app_err_2, r_app_err_2, i_app_err_2, z_app_err_2, u_abs_1, g_abs_1, r_abs_1, i_abs_1, z_abs_1, u_abs_2, g_abs_2, r_abs_2, i_abs_2, z_abs_2,plate1, mjd1, fiberid1, plate2, mjd2, fiberid2,specz1,specz2,photoz1,photoz2,ra1,ra2,dec1,dec2,vote1,vote2,kmass1,kmass2,kmass_err1,kmass_err2'

; Print data to a csv file

for i = 0,n_elements(ra1) - 1 do begin
    printf,lun1,u_app_1[i],',', g_app_1[i],',', r_app_1[i],',', i_app_1[i],',', z_app_1[i],',', u_app_2[i],',', g_app_2[i],',', r_app_2[i],',', i_app_2[i],',', z_app_2[i],',', u_app_err_1[i],',', g_app_err_1[i],',', r_app_err_1[i],',', i_app_err_1[i],',', z_app_err_1[i],',', u_app_err_2[i],',', g_app_err_2[i],',', r_app_err_2[i],',', i_app_err_2[i],',', z_app_err_2[i],',', u_abs_1[i],',', g_abs_1[i],',', r_abs_1[i],',', i_abs_1[i],',', z_abs_1[i],',', u_abs_2[i],',', g_abs_2[i],',', r_abs_2[i],',', i_abs_2[i],',', z_abs_2[i],',',plate1[i],',', mjd1[i],',', fiberid1[i],',', plate2[i],',', mjd2[i],',', fiberid2[i],',',SPECZ1[i],',',SPECZ2[i],',',PHOTOZ1[i],',',PHOTOZ2[i],',',RA1[i],',',RA2[i],',',DEC1[i],',',DEC2[i],',',VOTE1[i],',',VOTE2[i],',',KMASS1[i],',',KMASS2[i],',',KMASS_ERR1[i],',',KMASS_ERR2[i]
endfor

free_lun,lun1

print,''

; Load second set of merger data (Galaxy Zoo votes) from IDL sav files
restore,merger_path+'/merger_catalogue.sav'
openw,lun2,merger_path+'/merger_catalogue.csv',/get_lun,width=200

; Print to csv file
printf,lun2,'# object1, object2, morph1, morph2, comments, stage'
for i = 0,n_elements(object1) - 1 do begin
    printf,lun2, OBJECT1[i], ',', OBJECT2[i], ',', MORPH1[i], ',', MORPH2[i], ',', COMMENTS[i], ',', STAGE[i]
endfor

print,'OBJECT1    ',n_elements(OBJECT1)
print,'OBJECT2    ',n_elements(OBJECT2)
print,'MORPH1    ',n_elements(MORPH1)
print,'MORPH2    ',n_elements(MORPH2)
print,'COMMENTS    ',n_elements(COMMENTS)
print,'STAGE    ',n_elements(STAGE)

free_lun,lun2

; After printing to CSV files, objects are indexed and merged in TOPCAT. 

end
