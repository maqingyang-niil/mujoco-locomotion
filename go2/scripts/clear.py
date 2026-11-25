from pathlib import Path
import shutil


GO2_ROOT=Path(__file__).resolve().parents[1]
TARGETS=(
    GO2_ROOT/"logs",
    GO2_ROOT/"models",
)


def clear_directory(directory:Path):
    directory.mkdir(parents=True,exist_ok=True)

    for item in directory.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def main():
    print("将永久删除以下目录中的全部内容：")

    for directory in TARGETS:
        print(f"{directory}")

    confirmation=input("输入yes:")

    if confirmation!="yes" or confirmation!="y":
        print("已取消，没有删除任何文件。")
        return

    for directory in TARGETS:
        clear_directory(directory)
        print(f"已清空{directory}")


if __name__=="__main__":
    main()