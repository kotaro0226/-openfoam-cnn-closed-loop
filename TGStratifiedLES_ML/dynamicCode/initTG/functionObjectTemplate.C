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

defineTypeNameAndDebug(initTGFunctionObject, 0);

addRemovableToRunTimeSelectionTable
(
    functionObject,
    initTGFunctionObject,
    dictionary
);


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

// dynamicCode:
// SHA1 = ced65db7d790d93e7ffdb41fa3979aea4cf697f7
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void initTG_ced65db7d790d93e7ffdb41fa3979aea4cf697f7(bool load)
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
Foam::initTGFunctionObject::mesh() const
{
    return refCast<const fvMesh>(obr_);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::
initTGFunctionObject::
initTGFunctionObject
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
initTGFunctionObject::
~initTGFunctionObject()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool
Foam::
initTGFunctionObject::read(const dictionary& dict)
{
    if (false)
    {
        printMessage("read initTG");
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool
Foam::
initTGFunctionObject::execute()
{
    if (false)
    {
        printMessage("execute initTG");
    }

//{{{ begin code
    #line 55 "/lustre/orion/scratch/whbrewer/cfd135/TGStratifiedLES/system/controlDict/functions/initTG"
Info << "Initializing Taylor–Green vortex fields..." << endl;

            const fvMesh& mesh = refCast<const fvMesh>(obr_);
            volVectorField& U     = const_cast<volVectorField&>(
                                        mesh.lookupObject<volVectorField>("U"));
            volScalarField& T     = const_cast<volScalarField&>(
                                        mesh.lookupObject<volScalarField>("T"));
            volScalarField& p_rgh = const_cast<volScalarField&>(
                                        mesh.lookupObject<volScalarField>("p_rgh"));

            const scalar U0   = 1.0;   // characteristic velocity
            const scalar rho0 = 1.0;   // reference density
            const scalar T0   = 300.0; // baseline temperature

            forAll(mesh.C(), i)
            {
                const point& pt = mesh.C()[i];
                scalar x = pt.x();
                scalar y = pt.y();
                scalar z = pt.z();

                // Taylor–Green velocity field
                U[i].x() =  U0 * sin(x) * cos(y) * cos(z);
                U[i].y() = -U0 * cos(x) * sin(y) * cos(z);
                U[i].z() = 0.0;

                // Pressure (hydrostatic-free form)
                scalar p = (rho0*U0*U0/16.0)
                          * (cos(2.0*x) + cos(2.0*y))
                          * (cos(2.0*z) + 2.0);
                p_rgh[i] = p;

                // Uniform temperature
                T[i] = T0;
            }

            // Apply BC corrections
            U.correctBoundaryConditions();
            p_rgh.correctBoundaryConditions();
            T.correctBoundaryConditions();

            Info << "Taylor–Green vortex initialization complete." << endl;
//}}} end code

    return true;
}


bool
Foam::
initTGFunctionObject::write()
{
    if (false)
    {
        printMessage("write initTG");
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool
Foam::
initTGFunctionObject::end()
{
    if (false)
    {
        printMessage("end initTG");
    }

//{{{ begin code
    
//}}} end code

    return true;
}


// ************************************************************************* //

