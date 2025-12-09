export default function NeurondLogo({ className = "w-8 h-8" }: { className?: string }) {
    return (
        <div className={`${className} rounded-lg overflow-hidden shadow-sm bg-white p-1 flex items-center justify-center`}>
            <img
                src="https://cdn.neurond.com/neurondasset/assets/image/icon/neurond-final-black.svg"
                alt="Neurond AI"
                className="w-full h-full object-contain"
            />
        </div>
    );
}
