pro addnoise,filein,band,exptime,nexp,fileout
;; read image

img=readfits(filein,hdr)
;; telescope and detector parameters
;diam=30.	; primary diameter
;sec=2.		; secondary diameter
;eff=0.331	; system efficiency
;nvega=2.02e10	; photons s^-1 m^-2 micron^-1
;dlam=0.26	; microns
;exptime=100.	; seconds
apix=0.005	; arcsec per pixel
gain=2.		; e-/adu
rn=15.		; read noise in e-
satlev=65535.	; saturation level in ADU
funres=0.0	; fraction unresolved light

;zp=2.5*alog10(nvega*dlam*eff*exptime*!pi*(diam/2-sec/2)^2)
;scale=10^(-0.4*(22-zp))
if band eq 'J' then scale=678536.45/630.96*exptime  ;;; from Bob's spreadsheet
if band eq 'H' then scale=807529/630.96*exptime  ;;; from Bob's spreadsheet
if band eq 'K' then scale=430284/630.96*exptime  ;;; from Bob's spreadsheet
;unres=total(img)*scale*funres
;imunres=nr/total(nr)*unres

;; calculate sky
;skym=16.2	; J-band
;skyc=10^(-0.4*(skym-zp))*apix^2
if band eq 'J' then skyc=5.6*exptime	;;; from Bob's spreadsheet
if band eq 'H' then skyc=(46.2+0.1)*exptime	;;; from Bob's spreadsheet
if band eq 'K' then skyc=(35.6+34.7)*exptime	;;; from Bob's spreadsheet

;nexp=100.
;; scale image and add sky and unresolved light
img1=img*scale+skyc;+imunres

;; calculate and add noise
sz=size(img1)
noise=randomn(s,sz(1),sz(2))*sqrt(img1) + randomn(s,sz(1),sz(2))*rn
img1=img1+noise/sqrt(nexp)

;; apply gain and sat. level
img1=img1/gain

writefits,fileout,img1<satlev,hdr


end

