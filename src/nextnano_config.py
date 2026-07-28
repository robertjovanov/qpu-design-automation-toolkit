import os

path_license = r"/Applications/nextnano/2025_12_17/licenses"
path_nextnano_output = r"/Users/robertjovanov/code/qpu-design-automation-toolkit/runs"
path_nextnano = r"/Applications/nextnano/2025_12_17"


def configure_nextnano(
    license_directory: str = path_license,
    output_directory: str = path_nextnano_output,
    nextnano_directory: str = path_nextnano,
    *,
    save: bool = True,
    config=None,
) -> None:
    if config is None:
        import nextnanopy as nn

        config = nn.config

    config.to_default()

    config.set("nextnano++", "outputdirectory", output_directory)
    config.set("nextnano3", "outputdirectory", output_directory)
    config.set("nextnano.NEGF", "outputdirectory", output_directory)
    config.set("nextnano.MSB", "outputdirectory", output_directory)

    config.set("nextnano3", "license", os.path.join(license_directory, "license.txt"))
    config.set("nextnano++", "license", os.path.join(license_directory, "license.txt"))
    config.set(
        "nextnano.NEGF",
        "license",
        os.path.join(license_directory, "License_nnNEGF.lic"),
    )

    config.set(
        "nextnano++",
        "exe",
        os.path.join(
            nextnano_directory,
            "nextnano++/bin/nextnano++_gcc_macOS_old",
        ),
    )
    config.set(
        "nextnano++",
        "database",
        os.path.join(nextnano_directory, "nextnano++/database/database.nnp"),
    )
    config.set(
        "nextnano3",
        "exe",
        os.path.join(
            nextnano_directory,
            "nextnano3/bin/nextnano3_gcc_macOS_old",
        ),
    )
    config.set(
        "nextnano3",
        "database",
        os.path.join(nextnano_directory, "nextnano3/database/database.nn3"),
    )
    config.set(
        "nextnano.NEGF",
        "exe",
        os.path.join(
            nextnano_directory,
            "nextnano.NEGF/bin/nextnano.NEGF_win.exe",
        ),
    )
    config.set(
        "nextnano.NEGF",
        "database",
        os.path.join(
            nextnano_directory,
            "nextnano.NEGF/database/Material_Database.in",
        ),
    )
    config.set(
        "nextnano.MSB",
        "database",
        os.path.join(nextnano_directory, "nextnano.MSB/database/materials.msb"),
    )

    if save:
        config.save()


def main() -> None:
    import nextnanopy as nn

    print(f"The nextnanopy config file is stored in: {nn.config.fullpath}")
    configure_nextnano(config=nn.config, save=True)
    print("The nextnanopy config file has been updated and saved.")


if __name__ == "__main__":
    main()
