volScalarField rAU("rAU", 1.0/UEqn.A());
surfaceScalarField rAUf("rAUf", fvc::interpolate(rAU));
volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p_rgh));

surfaceScalarField phig(-rAUf*ghf*fvc::snGrad(rhok)*mesh.magSf());

surfaceScalarField phiHbyA
(
    "phiHbyA",
    fvc::flux(HbyA)
  + MRF.zeroFilter(rAUf*fvc::ddtCorr(U, phi))
  + phig
);

MRF.makeRelative(phiHbyA);
constrainPressure(p_rgh, U, phiHbyA, rAUf, MRF);

