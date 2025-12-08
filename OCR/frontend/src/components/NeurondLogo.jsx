// Neurond AI Logo from official CDN with container box
export default function NeurondLogo({ className = "w-8 h-8", size = "default" }) {
    // Size presets
    const sizes = {
        small: "w-8 h-8",
        default: "w-10 h-10",
        large: "w-12 h-12"
    };

    const containerSize = sizes[size] || className;

    return (
        <div
            className={`${containerSize} rounded-lg bg-white flex items-center justify-center p-1.5`}
            style={{
                boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
            }}
        >
            <img
                src="https://cdn.neurond.com/neurondasset/assets/image/icon/neurond-final-black.svg"
                alt="Neurond AI"
                className="w-full h-full object-contain"
            />
        </div>
    );
}
