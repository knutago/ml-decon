pro addnoise20,filein,band,exptime,nexp,fileout
;; read image

img=readfits(filein,hdr)
gain=1.		; e-/adu
rn=15.		; read noise in e-
satlev=1e5	; saturation level in ADU
funres=0.0	; fraction unresolved light

if band eq 'J' then scale=301572.*10^(-0.4*(25.-15))*exptime
if band eq 'H' then scale=358902.*10^(-0.4*(25.-15))*exptime
if band eq 'K' then scale=191237.*10^(-0.4*(25.-15))*exptime
;unres=total(img)*scale*funres
;imunres=nr/total(nr)*unres

;; calculate sky
;skym=16.2	; J-band
;skyc=10^(-0.4*(skym-zp))*apix^2
if band eq 'J' then skyc=3.8*exptime	;;; from Bob's spreadsheet
if band eq 'H' then skyc=(31.6+0.0)*exptime	;;; from Bob's spreadsheet
if band eq 'K' then skyc=(24.3+23.7)*exptime	;;; from Bob's spreadsheet

;nexp=100.
;; scale image and add sky and unresolved light
img1=img*scale+skyc;+imunres

;; calculate and add noise
sz=size(img1)
noise=randomn(s,sz(1),sz(2))*sqrt(img1) + randomn(s,sz(1),sz(2))*rn
img1=img1+noise/sqrt(nexp)

;; apply gain and sat. level
img1=img1/gain
print,median(img1),round(max(img1)/satlev)
writefits,fileout,(img1/nexp)<(satlev),hdr


end

