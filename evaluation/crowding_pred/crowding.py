import numpy as np
from scipy.interpolate import interp1d, interpn


def compCrowdError(magVector, lumFunc, seeing, lumAreaArcsec=3600.**2, singleMag=None):
        """
        Compute the crowding error for each observation

        Parameters
        ----------
        magVector : np.array
            Stellar magnitudes.
        lumFunc : np.array
            Stellar luminosity function, in counts over the area lumAreaArcsec.
        seeing : float
            The best seeing conditions. Assuming forced-photometry can use the best seeing conditions
            to help with confusion errors.
        lumAreaArcsec : float
            Area over which lumFunc counts are measured, in arcsec^2. Default is
            one square degree (3600^2), so lumFunc is then per square degree.
        singleMag : float (None)
            If singleMag is None, the crowding error is calculated for each mag in magVector. If
            singleMag is a float, the crowding error is interpolated to that single value.

        Returns
        -------
        np.array
            Magnitude uncertainties.


        Equation from Olsen, Blum, & Rigaut 2003, AJ, 126, 452
        """

        lumVector = 10**(-0.4*magVector)
        coeff=np.sqrt(np.pi/lumAreaArcsec)*seeing/2.
        myIntegral = (np.add.accumulate((lumVector**2*lumFunc)[::-1]))[::-1]
        temp = np.sqrt(myIntegral)/lumVector
        if singleMag is not None:
            interp = interp1d(magVector, temp)
            temp = interp(singleMag)

        crowdError = 1.086*coeff*temp

        return crowdError
    
def compColorCrowdError(magVector1, magVector2, lumFunc12, seeing1, seeing2, lumAreaArcsec=3600.**2, singleMag1=None, singleMag2=None):
        """
        Compute the crowding error for each observation

        Parameters
        ----------
        magVector1 : np.array
            Stellar magnitudes in band 1.
        magVector2 : np.array
            Stellar magnitudes in band 2.
        lumFunc12 : np.array (2D)
            Joint stellar luminosity function in bands 1 and 2, in counts over
            the area lumAreaArcsec.
        seeing1 : float
            Seeing in band 1.
        seeing2 : float
            Seeing in band 2.
        lumAreaArcsec : float
            Area over which lumFunc12 counts are measured, in arcsec^2. Default is
            one square degree (3600^2), so lumFunc12 is then per square degree.
        singleMag1 : float (None)
            If singleMag is None, the crowding error is calculated for each mag in magVector. If
            singleMag is a float, the crowding error is interpolated to that single value.

        Returns
        -------
        np.array
            Magnitude uncertainties.


        Equation from Olsen, Blum, & Rigaut 2003, AJ, 126, 452
        """

        lumVector1, lumVector2 = np.meshgrid(10**(-0.4*magVector1), 10**(-0.4*magVector2))

        coeff=np.sqrt(np.pi/lumAreaArcsec)*np.min([seeing1,seeing2])/2.
        innerInt = np.add.accumulate(np.flip(lumVector1*lumFunc12),axis=1)
        outerInt = np.add.accumulate(np.flip(lumVector2)*innerInt,axis=0)
        int2d = np.flip(outerInt)

        temp = np.sqrt(int2d)/np.sqrt(lumVector1*lumVector2)

        crowdError12 = 1.086*coeff*temp
        crowdError1 = compCrowdError(magVector1,np.sum(lumFunc12,axis=0),seeing1,lumAreaArcsec)
        crowdError2 = compCrowdError(magVector2,np.sum(lumFunc12,axis=1),seeing2,lumAreaArcsec)
        ce1, ce2 = np.meshgrid(crowdError1,crowdError2)
        
        colCrowdError = np.sqrt(ce1**2 + ce2**2 - 2*crowdError12**2)
        if singleMag1 is not None and singleMag2 is not None:
            interp = interp1d(magVector1, crowdError1)
            crowdError1 = interp(singleMag1)
            interp = interp1d(magVector2, crowdError2)
            crowdError2 = interp(singleMag2)
            #xvec,yvec = np.meshgrid(magVector1,magVector2)
            interp = interpn((magVector1,magVector2),colCrowdError,(singleMag1,singleMag2))
            colCrowdError = interp

        return crowdError1, crowdError2, colCrowdError
