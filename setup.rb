require 'atk_toolbox'


if not Console.has_command("python3")
    system "atk run jeff-hykin/install-python"
end

system "pip install opencv-python"
system "pip install numpy"