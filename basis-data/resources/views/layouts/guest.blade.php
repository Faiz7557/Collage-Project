<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Selamat Datang - Warung Padi</title>
    
    <!-- Bootstrap 5.3 untuk konsistensi dan upgrade tampilan -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Font Poppins untuk konsistensi -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Bootstrap Icons untuk icons modern -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    
    <!-- Custom CSS: Hanya upgrade untuk Navbar (glassmorphism, animasi, gradients) -->
    <style>
        :root {
            --primary-green: #28a745;
            --secondary-green: #20c997;
            --glass-bg: rgba(255, 255, 255, 0.25);
            --glass-border: rgba(255, 255, 255, 0.18);
            --shadow-light: 0 8px 32px rgba(0, 0, 0, 0.1);
            --shadow-heavy: 0 4px 30px rgba(0, 0, 0, 0.2);
        }
        
        body {
            /* Background handled in welcome.blade.php for specific sections or here globally */
            margin: 0;
            padding: 0;
            font-family: 'Poppins', sans-serif;
            overflow-x: hidden;
            background-color: #f8f9fa;
        }
        
        /* Navbar Upgrade: Brighter Glassmorphism independent of scroll */
        /* Navbar Upgrade: Brighter Glassmorphism independent of scroll */
        .navbar {
            background: rgba(255, 255, 255, 0.7); /* Translucent again for glassmorph */
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-bottom: 2px solid rgba(255, 255, 255, 0.5);
            padding: 1.5rem 0;
            transition: all 0.4s ease;
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1000;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }
        
        .navbar.scrolled {
            padding: 1.2rem 0; /* Still substantial on scroll */
            box-shadow: 0 4px 25px rgba(0,0,0,0.1);
            background: rgba(255, 255, 255, 0.98);
        }

        /* Force Nav Links to be Dark Green/Black for readability on the bright glass */
        .navbar .nav-link {
            color: #1a1a1a !important; /* Almost black for max contrast */
            font-weight: 700; /* Bolder text */
            margin: 0 20px; /* More spacing */
            transition: all 0.3s ease;
            font-size: 1.1rem; /* Larger font size */
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        
        .navbar.scrolled .nav-link {
            color: #2c3e50 !important;
        }
        
        .navbar-brand img {
            height: 45px;
            transition: transform 0.3s ease;
            filter: none; /* Always full color */
        }
        
        .nav-link:hover {
            opacity: 1;
            color: var(--primary-green) !important;
            transform: translateY(-1px);
        }
        
        /* Navbar Toggler - Dark by default now */
        .navbar-toggler-icon {
            filter: none; 
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 30 30'%3e%3cpath stroke='rgba%280, 0, 0, 0.7%29' stroke-linecap='round' stroke-miterlimit='10' stroke-width='2' d='M4 7h22M4 15h22M4 23h22'/%3e%3c/svg%3e");
        }
        
        .navbar-toggler-icon {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 30 30'%3e%3cpath stroke='rgba%28255, 255, 255, 0.8%29' stroke-linecap='round' stroke-miterlimit='10' stroke-width='2' d='M4 7h22M4 15h22M4 23h22'/%3e%3c/svg%3e");
        }

        /* User Greeting Section */
        .user-greeting {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #fff;
            font-weight: 500;
            background: var(--glass-bg);
            padding: 8px 15px;
            border-radius: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            transition: all 0.3s ease;
        }

        .user-greeting:hover {
            background: rgba(255, 255, 255, 0.35);
            transform: translateY(-1px);
            box-shadow: var(--shadow-light);
        }

        .user-avatar {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary-green), var(--secondary-green));
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-weight: 600;
            font-size: 1rem;
        }
        
        /* CSS Asli untuk Glass Card (tidak diubah, untuk child views jika diperlukan) */
        .glass-morphism-card {
            background: rgba(255, 255, 255, 0.4);
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 2rem;
        }
        
        /* Responsive untuk Navbar saja */
        @media (max-width: 992px) {
            .navbar .navbar-nav {
                text-align: center;
                margin-top: 1rem;
            }
            
            .nav-link {
                margin: 0.5rem 0;
            }
            
            .btn-signup, .btn-login {
                margin: 0.5rem;
                width: 100%;
                justify-content: center;
            }
        }
        
        @media (max-width: 576px) {
            .navbar-brand img {
                height: 40px;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-light" id="navbar">
        <div class="container">
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            
            <a class="navbar-brand fw-bold {{ (Request::routeIs('login') || Request::routeIs('register')) ? 'd-none' : '' }}" href="{{ Auth::check() ? route('dashboard') : '/' }}">
                <img src="{{ asset('images/logo.png') }}" alt="Warung Padi Logo">
            </a>

            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav mx-auto mb-2 mb-lg-0">
                    <li class="nav-item"><a class="nav-link" href="{{ Auth::check() ? route('dashboard') : route('welcome') }}"><i class="bi bi-house me-1"></i>HOME</a></li>
                    <li class="nav-item"><a class="nav-link" href="{{ route('market.index') }}"><i class="bi bi-shop me-1"></i>MARKET</a></li>
                    <li class="nav-item"><a class="nav-link" href="{{ route('about') }}"><i class="bi bi-info-circle me-1"></i>ABOUT US</a></li>
                    <li class="nav-item"><a class="nav-link" href="{{ route('contact-us') }}"><i class="bi bi-envelope me-1"></i>CONTACT US</a></li>
                </ul>
                @if(Auth::check())
                <div class="user-greeting">
                    <div class="user-avatar">
                        {{ substr(Auth::user()->name ?? 'User', 0, 1) }}
                    </div>
                    <span>Hello, {{ Auth::user()->name ?? 'User' }}</span>
                    <div class="dropdown position-relative">
                        <button class="btn btn-transparent p-0 ms-2 dropdown-toggle" type="button" id="userDropdown" data-bs-toggle="dropdown" aria-expanded="false">
                            <i class="bi bi-chevron-down"></i>
                        </button>
                        <ul class="dropdown-menu dropdown-menu-end p-0" style="background: white; border-radius: 10px; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);">
                            <div class="dropdown-header p-3 border-bottom">
                                <h6 class="mb-0 text-dark">{{ Auth::user()->name ?? 'User' }}</h6>
                                <small class="text-muted">{{ Auth::user()->email ?? '' }}</small>
                                <button type="button" class="btn-close position-absolute top-0 end-0 m-2" aria-label="Close" onclick="document.getElementById('userDropdown').click()"></button>
                            </div>
                            <li><a class="dropdown-item d-flex align-items-center py-2 px-3" href="{{ route('settings.index') }}"><i class="bi bi-gear me-2"></i>Pengaturan</a></li>
                            <li><hr class="dropdown-divider my-1"></li>
                            <li>
                                <form action="{{ url('/logout') }}" method="POST">
                                    @csrf
                                    <button type="submit" class="dropdown-item d-flex align-items-center py-2 px-3 text-danger"><i class="bi bi-box-arrow-right me-2"></i>Logout</button>
                                </form>
                            </li>
                        </ul>
                    </div>
                </div>
                @else
                <!-- Auth buttons removed from navbar -->
                @endif
            </div>
        </div>
    </nav>

    <!-- Main Content: Tidak diubah, tetap seperti asli -->
    <main class="py-4">
        @yield('content')
    </main>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Custom JS untuk Scroll Effect pada Navbar -->
    <script>
        // Navbar scroll effect
        window.addEventListener('scroll', function() {
            const navbar = document.getElementById('navbar');
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });
    </script>
</body>
</html>
