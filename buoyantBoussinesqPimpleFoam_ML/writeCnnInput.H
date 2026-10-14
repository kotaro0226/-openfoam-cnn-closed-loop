{
    static bool wroteCnnInputOnce = false;

    const char* writeCnnInputEnv = std::getenv("ML_WRITE_CNN_INPUT");
    const char* writeCnnInputOnceEnv = std::getenv("ML_WRITE_CNN_INPUT_ONCE");

    const bool writeCnnInput =
        writeCnnInputEnv != nullptr && word(writeCnnInputEnv) != "0";

    const bool writeOnce =
        writeCnnInputOnceEnv == nullptr || word(writeCnnInputOnceEnv) != "0";

    if (writeCnnInput && !(writeOnce && wroteCnnInputOnce))
    {
        const scalarField& rAUfValues = rAUf.primitiveField();
        const scalarField& phiHbyAValues = phiHbyA.primitiveField();
        const scalarField& divPhiHbyAValues = divPhiHbyA_f.primitiveField();

        const vectorField& cellCentres = mesh.C();
        const vectorField& faceCentres = mesh.Cf();
        const label nInternalFaces = mesh.nInternalFaces();

        const fileName outDir =
            runTime.path()/"postProcessing"/"cnnInput"/runTime.timeName();
        mkDir(outDir);

        OFstream os(outDir/"cnn_input_18ch.csv");
        os  << "cellId,validAllInternalFaces";
        const word faceNames[6] = {"xm", "xp", "ym", "yp", "zm", "zp"};
        for (label faceI = 0; faceI < 6; ++faceI)
        {
            const word& faceName = faceNames[faceI];
            os  << "," << faceName << "_rAUf"
                << "," << faceName << "_phiHbyA"
                << "," << faceName << "_divPhiHbyA_f";
        }
        os << nl;

        label nValid = 0;
        forAll(mesh.cells(), cellI)
        {
            scalar channels[18] = {};
            bool haveFace[6] = {};

            const cell& cFaces = mesh.cells()[cellI];
            forAll(cFaces, localFaceI)
            {
                const label faceI = cFaces[localFaceI];
                if (faceI >= nInternalFaces)
                {
                    continue;
                }

                const vector d = faceCentres[faceI] - cellCentres[cellI];
                const scalar ax = mag(d.x());
                const scalar ay = mag(d.y());
                const scalar az = mag(d.z());

                label faceSlot = -1;
                if (ax >= ay && ax >= az)
                {
                    faceSlot = d.x() < 0 ? 0 : 1;  // xm, xp
                }
                else if (ay >= ax && ay >= az)
                {
                    faceSlot = d.y() < 0 ? 2 : 3;  // ym, yp
                }
                else
                {
                    faceSlot = d.z() < 0 ? 4 : 5;  // zm, zp
                }

                const label ch = 3*faceSlot;
                channels[ch] = rAUfValues[faceI];
                channels[ch + 1] = phiHbyAValues[faceI];
                channels[ch + 2] = divPhiHbyAValues[faceI];
                haveFace[faceSlot] = true;
            }

            bool validAll = true;
            for (label faceSlot = 0; faceSlot < 6; ++faceSlot)
            {
                validAll = validAll && haveFace[faceSlot];
            }

            if (validAll)
            {
                ++nValid;
            }

            os << cellI << "," << label(validAll);
            for (label ch = 0; ch < 18; ++ch)
            {
                os << "," << channels[ch];
            }
            os << nl;
        }

        Info<< "[ML] Wrote CNN 18-channel input probe: "
            << outDir/"cnn_input_18ch.csv"
            << " validCells=" << nValid
            << " totalCells=" << mesh.nCells()
            << nl << endl;

        wroteCnnInputOnce = true;
    }
}
