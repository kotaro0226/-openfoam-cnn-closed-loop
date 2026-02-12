/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019-2021 OpenCFD Ltd.
    Copyright (C) YEAR AUTHOR, AFFILIATION
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "functionObjectTemplate.H"
#define namespaceFoam  // Suppress <using namespace Foam;>
#include "fvCFD.H"
#include "unitConversion.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(kineticEnergyFunctionObject, 0);

addRemovableToRunTimeSelectionTable
(
    functionObject,
    kineticEnergyFunctionObject,
    dictionary
);


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

// dynamicCode:
// SHA1 = 579e1e060326595c49cff313e6311a601caedb83
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void kineticEnergy_579e1e060326595c49cff313e6311a601caedb83(bool load)
{
    if (load)
    {
        // Code that can be explicitly executed after loading
    }
    else
    {
        // Code that can be explicitly executed before unloading
    }
}


// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode

} // End namespace Foam


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

const Foam::fvMesh&
Foam::kineticEnergyFunctionObject::mesh() const
{
    return refCast<const fvMesh>(obr_);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::
kineticEnergyFunctionObject::
kineticEnergyFunctionObject
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    functionObjects::regionFunctionObject(name, runTime, dict)
{
    read(dict);
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::
kineticEnergyFunctionObject::
~kineticEnergyFunctionObject()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool
Foam::
kineticEnergyFunctionObject::read(const dictionary& dict)
{
    if (false)
    {
        printMessage("read kineticEnergy");
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool
Foam::
kineticEnergyFunctionObject::execute()
{
    if (false)
    {
        printMessage("execute kineticEnergy");
    }

//{{{ begin code
    #line 129 "/lustre/orion/scratch/shunya/gen053/work/TGStratifiedLES_ML/system/controlDict/functions/kineticEnergy"
const fvMesh& m = mesh();

            // velocity
            const volVectorField& U = m.lookupObject<volVectorField>("U");

            // check for rho
            const bool haveRho = m.foundObject<volScalarField>("rho");

            // read dictionary
            const dictionary& d = dict();
            const bool useBoussinesq = Switch(d.lookupOrDefault("useBoussinesq", false));
            const scalar rhoRef = d.lookupOrDefault<scalar>("rhoRef", 1.0);
            const scalar beta   = d.lookupOrDefault<scalar>("beta", 0.0);
            const scalar TRef   = d.lookupOrDefault<scalar>("TRef", 300.0);
            const fileName csvPath = d.lookupOrDefault<fileName>("csvPath", "postProcessing/kineticEnergy/KE.csv");

            tmp<volScalarField> tKeDens;

            if (haveRho)
            {
                const volScalarField& rho = m.lookupObject<volScalarField>("rho");
                tKeDens = 0.5*rho*magSqr(U);
            }
            else if (useBoussinesq)
            {
                const volScalarField& T = m.lookupObject<volScalarField>("T");
                volScalarField rhoB
                (
                    IOobject("rhoB", time().timeName(), m, IOobject::NO_READ, IOobject::NO_WRITE),
                    rhoRef*(1.0 - beta*(T - TRef))
                );
                tKeDens = 0.5*rhoB*magSqr(U);
            }
            else
            {
                tKeDens = 0.5*rhoRef*magSqr(U);
            }

            const volScalarField& keDens = tKeDens();

            const scalarField& V = m.V();
            const scalar KE_total = gSum( keDens.internalField()*V );
            const scalar V_total = gSum(V);
            const scalar KE_avg  = KE_total / V_total;

            Info<< "KE_total = " << KE_total << " J,  KE_volAvg = " << KE_avg << " J/m^3" << nl << endl;

            // ensure directory exists
            mkDir(csvPath.path());

            // append to CSV
            OFstream os(csvPath, IOstreamOption(), IOstreamOption::APPEND);

            if (time().timeName() == "0")
            {
                os << "time,KE_total[J],KE_volAvg[J/m3]\n";
            }

            os << time().timeName() << "," << KE_total << "," << KE_avg << "\n";
//}}} end code

    return true;
}


bool
Foam::
kineticEnergyFunctionObject::write()
{
    if (false)
    {
        printMessage("write kineticEnergy");
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool
Foam::
kineticEnergyFunctionObject::end()
{
    if (false)
    {
        printMessage("end kineticEnergy");
    }

//{{{ begin code
    
//}}} end code

    return true;
}


// ************************************************************************* //

